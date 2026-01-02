import os
import gradio as gr


# Load API Key (Cloud Run environment variable)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Lazy RAG Initialization (for Cloud Run startup timeout)

qa_chain = None
retriever = None

def load_rag():
    """
    Loads the RAG pipeline only on first request.
    Prevents Cloud Run startup timeout.
    """
    global qa_chain, retriever
    if qa_chain is not None:
        return qa_chain, retriever

    print("Checking for PDF:", os.path.exists("PepsiCo_Global_Code_of_Conduct.pdf"))

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
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    print("RAG pipeline loaded successfully.")
    return qa_chain, retriever

# Fallback text

FALLBACK_TEXT = """
This question may not be explicitly covered in the PepsiCo Global Code of Conduct.

However, the Code emphasizes ethical behavior, legal compliance, anti-bribery and anti-corruption standards,
conflict-of-interest prevention, accurate reporting, workplace respect, human rights, safety,
and integrity in all business dealings.
"""

# Structured Answer Function (Block Answer Mode)

def answer_question(question, show_sources):
    if not question or question.strip() == "":
        return "Please enter a valid question."

    try:
        chain, retriever = load_rag()
        response = chain.invoke({"query": question})

        answer = response.get("result", "").strip()
        sources = response.get("source_documents", [])

        if not answer:
            answer = FALLBACK_TEXT

        first_sentence = answer.split(".")[0].strip()

        structured = f"""### **Key Rule**
{first_sentence}.

### **Required Action**
{answer}

### **Risk if Ignored**
Violating this policy may result in disciplinary action, reputational damage, or legal consequences.
"""

        if show_sources and sources:
            structured += "\n### **Sources Used**\n"
            for src in sources:
                structured += f"- Page {src.metadata.get('page', 'N/A')}\n"

        return structured

    except Exception as e:
        return f"Error: {str(e)}"

# PepsiCo UI Styling

PRIMARY_BLUE = "#005CB4"
SECONDARY_RED = "#E41E2B"
LIGHT_GRAY = "#F5F5F5"

EXAMPLE_QUESTIONS = [
    "What is considered a conflict of interest under the PepsiCo Code?",
    "How does PepsiCo address bribery and anti-corruption globally?",
    "How do I report a Code of Conduct violation at PepsiCo?",
    "What protections exist for employees who speak up?",    
    "What is PepsiCo’s policy on accepting gifts or entertainment?",    
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

# Gradio UI (Block Answer Mode)
# ---------------------------------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="red",
        neutral_hue="gray"
    ),
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
    """
) as demo:

    gr.HTML("""
        <div class="pepsico-header">
            PepsiCo Global Code of Conduct Assistant
        </div>
    """)

    gr.Markdown("Select a question from the list **or** type your own below.")

    with gr.Row():
        dropdown = gr.Dropdown(
            choices=EXAMPLE_QUESTIONS,
            label="Select a Question",
            interactive=True
        )
        user_input = gr.Textbox(
            label="Your Question",
            placeholder="Type your question or select one above..."
        )

    dropdown.change(load_question, dropdown, user_input)

    show_sources = gr.Checkbox(label="Show sources", value=False)

    ask_button = gr.Button("Ask", variant="primary")

    output_box = gr.Markdown(label="Answer")

    ask_button.click(
        answer_question,
        inputs=[user_input, show_sources],
        outputs=output_box
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
