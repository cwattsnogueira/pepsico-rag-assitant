import os
import re
import gradio as gr

# Load API Key
# ---------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Section hierarchy (exact titles from the PepsiCo PDF)
# ---------------------------------------------------------
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

PARENT_SECTIONS = list(SECTION_HIERARCHY.keys())

# Build child -> parent mapping, and include parents mapping to themselves
CHILD_TO_PARENT = {}
ALL_TITLES = set()

for parent, children in SECTION_HIERARCHY.items():
    CHILD_TO_PARENT[parent] = parent
    ALL_TITLES.add(parent)
    for child in children:
        CHILD_TO_PARENT[child] = parent
        ALL_TITLES.add(child)

# Match chunk to closest section title 
# ---------------------------------------------------------
def match_section_titles(text: str):
    """
    Returns (parent_section, child_section or None) based on keyword overlap
    with the known section titles from the official hierarchy.
    """
    text_lower = text.lower()
    if not text_lower.strip():
        return None, None

    best_title = None
    best_score = 0

    for title in ALL_TITLES:
        title_lower = title.lower()
        title_words = set(title_lower.split())
        overlap = sum(1 for w in title_words if w in text_lower)

        # Prefer exact phrase matches slightly
        if title_lower in text_lower:
            overlap += len(title_words)

        if overlap > best_score:
            best_score = overlap
            best_title = title

    if not best_title or best_score == 0:
        return None, None

    parent = CHILD_TO_PARENT.get(best_title)
    if parent and best_title in SECTION_HIERARCHY.get(parent, []):
        # best_title is a child
        return parent, best_title
    else:
        # best_title is a parent
        return best_title, None

# Lazy RAG Initialization
# ---------------------------------------------------------
qa_chain = None
retriever = None


def load_rag():
    global qa_chain, retriever
    if qa_chain is not None:
        return qa_chain, retriever

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    PDF_PATH = "PepsiCo_Global_Code_of_Conduct.pdf"
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # Attach parent/child section metadata
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

    from langchain.chains import RetrievalQA as LC_RetrievalQA

    qa_chain = LC_RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )

    return qa_chain, retriever

# Fallback Text
# ---------------------------------------------------------
FALLBACK_TEXT = """
This question may not be explicitly covered in the PepsiCo Global Code of Conduct.

However, the Code emphasizes ethical behavior, legal compliance, anti-bribery and anti-corruption standards,
conflict-of-interest prevention, accurate reporting, workplace respect, human rights, safety,
and integrity in all business dealings.
""".strip()

# Extract Key Rule (bullet-first, always ends with a period)
# ---------------------------------------------------------
def extract_key_rule(answer: str) -> str:
    """
    Extracts the first meaningful rule from the answer.
    Priority:
      1) First bullet/numbered item.
      2) First full sentence.
    Ensures the result ends with a period.
    """

    # 1) Try to extract from bullet/numbered lines
    lines = answer.splitlines()
    for line in lines:
        m = re.match(r"\s*(?:[-•*]|\d+[.)-])\s+(.*\S)", line)
        if m:
            rule = m.group(1).strip()
            if not rule.endswith((".", "!", "?")):
                rule += "."
            return rule

    # 2) Fallback: first full sentence
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if sentences:
        rule = sentences[0]
        if not rule.endswith((".", "!", "?")):
            rule += "."
        return rule

    return "Follow the PepsiCo Global Code of Conduct."



# Structured Answer Function
# ---------------------------------------------------------
def answer_question(question, show_sources):
    if not question or question.strip() == "":
        return "Please enter a valid question."

    try:
        chain, _ = load_rag()
        response = chain.invoke({"query": question})

        answer = response.get("result", "").strip()
        sources = response.get("source_documents", [])

        if not answer:
            answer = FALLBACK_TEXT

        key_rule = extract_key_rule(answer)

        structured = f"""### **Key Rule**
{key_rule}

### **Required Action**
{answer}

### **Risk if Ignored**
Violating this policy may result in disciplinary action, reputational damage, or legal consequences.
"""

        if show_sources and sources:
            structured += "\n### **Sources Used**\n"
            structured += "PepsiCo Global Code of Conduct\n"

            seen = set()
            for src in sources:
                parent = src.metadata.get("parent_section")
                child = src.metadata.get("child_section")

                if not parent and not child:
                    key = ("Relevant Section of the Code", None)
                    if key not in seen:
                        structured += "- Relevant Section of the Code\n"
                        seen.add(key)
                    continue

                key = (parent, child)
                if key in seen:
                    continue
                seen.add(key)

                if parent and child:
                    structured += f"- {parent} \u2192 {child}\n"
                elif parent:
                    structured += f"- {parent}\n"
                else:
                    structured += "- Relevant Section of the Code\n"

        return structured

    except Exception as e:
        return f"Error: {str(e)}"

# UI Styling
# ---------------------------------------------------------
PRIMARY_BLUE = "#005CB4"
SECONDARY_RED = "#E41E2B"
LIGHT_GRAY = "#F5F5F5"

EXAMPLE_QUESTIONS = [
    "Pepsico Mission and Vision",
    "What is considered a conflict of interest under the PepsiCo Code?",
    "How do I report a Code of Conduct violation at PepsiCo?",
    "What protections exist for employees who speak up?",
    "What is considered a conflict of interest under the PepsiCo Code?",
    "What is PepsiCo’s policy on accepting gifts or entertainment?",
    "How does PepsiCo address bribery and anti-corruption globally?",
    "What are the rules regarding accurate financial reporting?",
    "How does PepsiCo protect human rights in the workplace?",
    "What is the policy on harassment and discrimination?",
    "How should employees use social media responsibly?",
    "What are PepsiCo’s expectations regarding data privacy and confidential information?",
    "What should I do if I witness unsafe behavior in the workplace?",
    "How does PepsiCo handle environmental sustainability responsibilities?",
    "What is the policy on interacting with government officials?",
    "How should I respond if a coworker violates the Code?",
]


def load_question(q):
    return q

# Gradio UI
# ---------------------------------------------------------
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

    gr.HTML(
        """
        <div class="pepsico-header">
            PepsiCo Global Code of Conduct Assistant
        </div>
        """
    )

    gr.Markdown("Select a question from the list **or** type your own below.")

    with gr.Row():
        dropdown = gr.Dropdown(
            choices=EXAMPLE_QUESTIONS,
            label="Select a Question",
            interactive=True,
        )
        user_input = gr.Textbox(
            label="Your Question",
            placeholder="Type your question or select one above...",
        )

    dropdown.change(load_question, dropdown, user_input)

    show_sources = gr.Checkbox(label="Show sources", value=False)
    ask_button = gr.Button("Ask", variant="primary")
    output_box = gr.Markdown(label="Answer")

    ask_button.click(
        answer_question,
        inputs=[user_input, show_sources],
        outputs=output_box,
    )


# Cloud Run Port Logic
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Gradio on port {port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_api=False,
        quiet=True,
    )
