import os
import re
import gradio as gr


# Load API Key
# ---------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



# Lazy RAG Initialization
# ---------------------------------------------------------
qa_chain = None
retriever = None



# Improved Section Title Extractor
# ---------------------------------------------------------
def extract_section_title(text: str) -> str:
    """
    Extracts a meaningful section title using robust heuristics.
    Avoids garbage like 'Ha', 'F', or partial words.
    """
    lines = text.split("\n")
    candidates = []

    for line in lines:
        stripped = line.strip()

        # Skip empty or tiny lines
        if len(stripped) < 4:
            continue

        # Skip lines with punctuation or numbers
        if re.search(r"[\d\.\,\:\;\(\)]", stripped):
            continue

        # ALL CAPS headings
        if stripped.isupper() and 4 <= len(stripped.split()) <= 10:
            candidates.append(stripped)
            continue

        # Title Case headings
        if re.match(r"^[A-Z][a-z]+(\s[A-Z][a-z]+)*$", stripped):
            candidates.append(stripped)

    # Prefer the last heading (closest to content)
    if candidates:
        return candidates[-1]

    return "Relevant Section of the Code"



# Load RAG
# ---------------------------------------------------------
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
    print("Checking for PDF:", os.path.exists(PDF_PATH))

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    # Attach section metadata
    for chunk in chunks:
        section = extract_section_title(chunk.page_content)
        chunk.metadata["section"] = section

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

    print("RAG pipeline loaded successfully.")
    return qa_chain, retriever



# Fallback text
# ---------------------------------------------------------
FALLBACK_TEXT = """
This question may not be explicitly covered in the PepsiCo Global Code of Conduct.

However, the Code emphasizes ethical behavior, legal compliance, anti-bribery and anti-corruption standards,
conflict-of-interest prevention, accurate reporting, workplace respect, human rights, safety,
and integrity in all business dealings.
""".strip()



# Improved Key Rule Extraction
# ---------------------------------------------------------
def extract_key_rule(answer: str) -> str:
    """
    Extracts the first meaningful rule from the answer.
    Prefers bullet points. Falls back to first full sentence.
    """
    # Look for bullet points
    bullets = re.findall(r"[-•]\s*(.*)", answer)
    if bullets:
        return bullets[0].strip()

    # Fallback: first full sentence
    sentences = answer.split(".")
    if sentences:
        return sentences[0].strip()

    return "Follow the PepsiCo Code of Conduct."



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
            for src in sources:
                section = src.metadata.get("section", "Relevant Section of the Code")
                structured += f"- Section: {section}\n"

        return structured

    except Exception as e:
        return f"Error: {str(e)}"



# UI Styling
# ---------------------------------------------------------
PRIMARY_BLUE = "#005CB4"
SECONDARY_RED = "#E41E2B"
LIGHT_GRAY = "#F5F5F5"

EXAMPLE_QUESTIONS = [
    "What is considered a conflict of interest under the PepsiCo Code?",
    "How do I report a Code of Conduct violation at PepsiCo?",
    "What protections exist for employees who speak up?",    
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

    gr.HTML("""
        <div class="pepsico-header">
            PepsiCo Global Code of Conduct Assistant
        </div>
    """)

    gr.Markdown("Select a question from the list **or** type your own below.")

    with gr.Row():
        dropdown = gr.Dropdown(choices=EXAMPLE_QUESTIONS, label="Select a Question", interactive=True)
        user_input = gr.Textbox(label="Your Question", placeholder="Type your question or select one above...")

    dropdown.change(load_question, dropdown, user_input)

    show_sources = gr.Checkbox(label="Show sources", value=False)
    ask_button = gr.Button("Ask", variant="primary")
    output_box = gr.Markdown(label="Answer")

    ask_button.click(answer_question, inputs=[user_input, show_sources], outputs=output_box)



# Cloud Run Port Logic
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Gradio on port {port}")
    demo.launch(server_name="0.0.0.0", server_port=port, show_api=False, quiet=True)
