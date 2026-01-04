import os
import re
import gradio as gr

# Load API Key

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Official PepsiCo Section Hierarchy (exact titles)

SECTION_HIERARCHY = {
    "Introduction": [
        "Our Mission and Vision",
    ],
    "Act with Integrity": [
        "Why Do We Have a Code of Conduct?",
        "Who is Responsible for Our Code?",
        "What is My Personal Responsibility?",
        "How to Act with Integrity",
        "Lead by Example",
        "How Can I Seek Guidance and Report Violations?",
        "Speak Up Hotline",
        "Investigating Misconduct and Disciplinary Action",
        "Retaliation is Prohibited",
    ],
    "In Our Workplace": [
        "Inclusion for Growth",
        "Human Rights",
        "Anti-Discrimination/Anti-Harassment",
        "Environment, Health and Safety",
        "Substance Abuse",
        "Anti-Violence",
    ],
    "In Our Marketplace": [
        "Our Consumers",
        "Food Safety and Product Quality",
        "Responsible Marketing",
        "Our Customers",
        "Our Suppliers",
        "Fair Competition",
        "Anti-Bribery",
        "Identifying Government Officials",
        "Business Gifts",
        "Anti-Money Laundering",
        "International Sanctions and Trade Controls",
    ],
    "In Business": [
        "Maintaining Accurate Business Records",
        "Records Retention",
        "Financial Accuracy",
        "Financial Disclosures and Audits",
        "Privacy",
        "Artificial Intelligence",
        "Proper Use and Protection of Company Resources",
        "Physical Property and Financial Resources",
        "Electronic Assets",
        "Intellectual Property",
        "Protecting PepsiCo Information",
        "Insider Trading is Prohibited",
        "Conflicts of Interest",
        "Communicating with the Public",
        "Public Speaking and Press Inquiries",
        "Social Media",
    ],
    "In Our World": [
        "pep+ (PepsiCo Positive)",
        "The PepsiCo Foundation",
        "Be a Good Citizen",
        "Political Activities",
    ],
    "Resources": [
        "PepsiCo Global Compliance & Ethics Department",
        "PepsiCo Law Department",
    ],
}

# Build child -> parent mapping and set of all titles
CHILD_TO_PARENT = {}
ALL_TITLES = set()

for parent, children in SECTION_HIERARCHY.items():
    CHILD_TO_PARENT[parent] = parent
    ALL_TITLES.add(parent)
    for child in children:
        CHILD_TO_PARENT[child] = parent
        ALL_TITLES.add(child)

# Caches to reduce LLM calls

KEY_RULE_CACHE = {}    # key: answer str -> value: key_rule str
SUMMARY_CACHE = {}     # key: answer str -> value: summary str


# Match chunk to closest section title

def match_section_titles(text: str):
    text_lower = text.lower()
    best_title = None
    best_score = 0

    for title in ALL_TITLES:
        title_lower = title.lower()
        words = title_lower.split()
        overlap = sum(1 for w in words if w in text_lower)

        if title_lower in text_lower:
            overlap += len(words)

        if overlap > best_score:
            best_score = overlap
            best_title = title

    if not best_title or best_score == 0:
        return None, None

    parent = CHILD_TO_PARENT.get(best_title)
    if parent and best_title in SECTION_HIERARCHY.get(parent, []):
        return parent, best_title
    return best_title, None

# Classify question as POLICY or INFORMATIVE

POLICY_KEYWORDS = [
    "policy", "rule", "allowed", "not allowed", "conflict",
    "harassment", "discrimination", "bribery", "gifts",
    "report", "violation", "safety", "compliance",
    "responsibilities", "should i", "must i", "required",
]

INFORMATIVE_KEYWORDS = [
    "mission", "vision", "values", "what is", "explain",
    "describe", "meaning", "pep+", "hotline", "foundation",
    "inclusion", "speak up", "our consumers", "our suppliers",
]


def classify_question(q: str) -> str:
    q_lower = q.lower()

    if any(k in q_lower for k in POLICY_KEYWORDS):
        return "policy"

    if any(k in q_lower for k in INFORMATIVE_KEYWORDS):
        return "informative"

    # Default: safer to treat as informative
    return "informative"

# Lazy RAG Initialization

qa_chain = None
retriever = None
SUMMARY_LLM = None  # shared for summary and key rule


def get_summary_llm():
    global SUMMARY_LLM
    if SUMMARY_LLM is None:
        from langchain_openai import ChatOpenAI
        SUMMARY_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return SUMMARY_LLM


def load_rag():
    global qa_chain, retriever
    if qa_chain is not None:
        return qa_chain, retriever

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader("PepsiCo_Global_Code_of_Conduct.pdf")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        parent, child = match_section_titles(chunk.page_content)
        if parent:
            chunk.metadata["parent_section"] = parent
        if child:
            chunk.metadata["child_section"] = child

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    return qa_chain, retriever
    

# Generate Key Rule (using LLM + cache)
def get_key_rule(answer: str) -> str:
    if not answer.strip():
        return "Follow the PepsiCo Global Code of Conduct."

    if answer in KEY_RULE_CACHE:
        return KEY_RULE_CACHE[answer]

    llm = get_summary_llm()
    prompt = f"""
    Summarize the core rule from the following text in one short, direct sentence.
    Do NOT list multiple items. Do NOT include examples.
    Answer as a clear rule starting with a verb (e.g., "Disclose...", "Avoid...", "Do not...").
    Text:
    {answer}
    """
    resp = llm.invoke(prompt)
    rule = resp.content.strip()
    if not rule.endswith((".", "!", "?")):
        rule += "."
    KEY_RULE_CACHE[answer] = rule
    return rule
    
# Generate Summary (using LLM + cache)
def get_summary(answer: str) -> str:
    if not answer.strip():
        return "This topic is described in the PepsiCo Global Code of Conduct."

    if answer in SUMMARY_CACHE:
        return SUMMARY_CACHE[answer]

    llm = get_summary_llm()
    prompt = f"""
    Summarize the following text in one short, high-level sentence.
    Do NOT repeat all details. Focus on the main idea.
    Text:
    {answer}
    """
    resp = llm.invoke(prompt)
    summary = resp.content.strip()
    if not summary.endswith((".", "!", "?")):
        summary += "."
    SUMMARY_CACHE[answer] = summary
    return summary

    
# Build Policy Answer (Key Rule + Required Action)
def build_policy_answer(answer: str, sources, show_sources: bool) -> str:
    key_rule = get_key_rule(answer)

    out = f"""### **Key Rule**
{key_rule}

### **Required Action**
{answer}
"""

    if show_sources:
        out += "\n### **Sources Used**\n"
        out += "PepsiCo Global Code of Conduct\n"

        seen = set()
        for src in sources or []:
            parent = src.metadata.get("parent_section")
            child = src.metadata.get("child_section")
            key = (parent, child)
            if key in seen:
                continue
            seen.add(key)

            if parent and child:
                out += f"- {parent} → {child}\n"
            elif parent:
                out += f"- {parent}\n"
            else:
                out += "- Relevant Section of the Code\n"

    return out


# Build Informative Answer (Summary + Details)
def build_informative_answer(answer: str, sources, show_sources: bool) -> str:
    summary = get_summary(answer)
    details = answer

    out = f"""### **Summary**
{summary}

### **Details**
{details}
"""

    if show_sources:
        out += "\n### **Sources Used**\n"
        out += "PepsiCo Global Code of Conduct\n"

        seen = set()
        for src in sources or []:
            parent = src.metadata.get("parent_section")
            child = src.metadata.get("child_section")
            key = (parent, child)
            if key in seen:
                continue
            seen.add(key)

            if parent and child:
                out += f"- {parent} → {child}\n"
            elif parent:
                out += f"- {parent}\n"
            else:
                out += "- Relevant Section of the Code\n"

    return out

# Main Answer Function
FALLBACK_TEXT = """
This question may not be explicitly covered in the PepsiCo Global Code of Conduct.

However, the Code emphasizes ethical behavior, legal compliance, anti-bribery and anti-corruption standards,
conflict-of-interest prevention, accurate reporting, workplace respect, human rights, safety,
and integrity in all business dealings.
""".strip()


def answer_question(question, show_sources):
    if not question or not question.strip():
        return "Please enter a valid question."

    chain, _ = load_rag()
    response = chain.invoke({"query": question})

    answer = response.get("result", "").strip()
    sources = response.get("source_documents", [])

    if not answer:
        answer = FALLBACK_TEXT

    qtype = classify_question(question)

    if qtype == "policy":
        return build_policy_answer(answer, sources, show_sources)

    return build_informative_answer(answer, sources, show_sources)

# UI Styling and Question Lists
PRIMARY_BLUE = "#005CB4"
SECONDARY_RED = "#E41E2B"
LIGHT_GRAY = "#F5F5F5"

INFORMATIVE_QUESTIONS = [
    "PepsiCo Mission and Vision",
    "What is pep+?",
    "What is the Speak Up Hotline?",
    "What is Inclusion for Growth?",
    "What is the PepsiCo Foundation?",
]

POLICY_QUESTIONS = [
    "What is considered a conflict of interest under the PepsiCo Code?",
    "How do I report a Code of Conduct violation at PepsiCo?",
    "What protections exist for employees who speak up?",
    "What is PepsiCo’s policy on accepting gifts or entertainment?",
    "How does PepsiCo address bribery and anti-corruption globally?",
    "What are the rules regarding accurate financial reporting?",
    "What is the policy on harassment and discrimination?",
    "How should employees use social media responsibly?",
    "What are PepsiCo’s expectations regarding data privacy and confidential information?",
    "What should I do if I witness unsafe behavior in the workplace?",
    "What is the policy on interacting with government officials?",
    "How should I respond if a coworker violates the Code?",
]


def load_question(q):
    return q

# Gradio UI
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="red", neutral_hue="gray"),
    css=f"""
        body {{
            background-color: {LIGHT_GRAY};
        }}
        .pepsico-header {{
            background: linear-gradient(90deg, {PRIMARY_BLUE}, {SECONDARY_RED});
            padding: 20px;
            border-radius: 8px;
            color: white;
            font-size: 26px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }}
        .gradio-container {{
            max-width: 900px !important;
            margin: auto;
        }}
    """,
) as demo:

    gr.HTML("""
        <div class="pepsico-header">
            PepsiCo Global Code of Conduct Assistant (Use‑Case Prototype)
        </div>
    """)

    gr.Markdown("Choose from suggested questions or type your own below.")

    with gr.Row():
        dropdown_info = gr.Dropdown(
            choices=INFORMATIVE_QUESTIONS,
            label="Select a Question (Informative)",
            interactive=True,
        )
        dropdown_policy = gr.Dropdown(
            choices=POLICY_QUESTIONS,
            label="Select a Question (Policy)",
            interactive=True,
        )

    user_input = gr.Textbox(
        label="Your Question",
        placeholder="Type your question here...",
    )

    dropdown_info.change(load_question, dropdown_info, user_input)
    dropdown_policy.change(load_question, dropdown_policy, user_input)

    show_sources = gr.Checkbox(label="Show sources", value=True)
    ask_button = gr.Button("Ask", variant="primary")
    output_box = gr.Markdown(label="Answer")

    ask_button.click(
        answer_question,
        inputs=[user_input, show_sources],
        outputs=output_box,
    )

# Cloud Run Port Logic
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Gradio on port {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_api=False,
        quiet=True,
    )
