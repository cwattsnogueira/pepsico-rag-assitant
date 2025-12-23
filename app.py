import os
import gradio as gr

# ---------------------------------------------------------
# Load API Key (Cloud Run environment variable)
# ---------------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ---------------------------------------------------------
# Lazy RAG Initialization (Fix for Cloud Run startup timeout)
# ---------------------------------------------------------
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

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    PDF_PATH = "PBG_English_3_28.pdf"

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
        model="gpt-3.5-turbo",
        temperature=0.2
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain, retriever


# ---------------------------------------------------------
# Fallback text
# ---------------------------------------------------------
FALLBACK_TEXT = """
This question may not be explicitly covered in the Pepsi Bottling Group Worldwide Code of Conduct.

However, the Code emphasizes ethical behavior, legal compliance, anti-bribery rules,
conflict-of-interest prevention, accurate reporting, workplace respect, safety,
and integrity in all business dealings.
"""


# ---------------------------------------------------------
# Structured Answer Function + Source Toggle + Chat History
# ---------------------------------------------------------
def answer_question(question, show_sources, history):
    if not question or question.strip() == "":
        return history + [[question, "Please enter a valid question."]]

    try:
        chain, retriever = load_rag()
        response = chain.invoke({"query": question})

        answer = response.get("result", "").strip()
        sources = response.get("source_documents", [])

        if not answer:
            answer = FALLBACK_TEXT

        # Structured Compliance Format
        structured = f"""**Key Rule:**  
{answer.split('.')[0].strip()}.

**Required Action:**  
{answer}

**Risk if Ignored:**  
Violating this policy may result in disciplinary action, reputational damage, or legal consequences.
"""

        # Optional Source Citations
        if show_sources and sources:
            structured += "\n\n**Sources Used:**\n"
            for i, src in enumerate(sources, start=1):
                structured += f"- Source {i}: Page {src.metadata.get('page', 'N/A')}\n"

        history = history + [[question, structured]]
        return history

    except Exception as e:
        return history + [[question, f"Error: {str(e)}"]]


# ---------------------------------------------------------
# PepsiCo UI Styling
# ---------------------------------------------------------
PRIMARY_BLUE = "#005CB4"
SECONDARY_RED = "#E41E2B"
LIGHT_GRAY = "#F5F5F5"

EXAMPLE_QUESTIONS = [
    "What should I do if I suspect an ethics violation?",
    "How do I report misconduct anonymously?",
    "What is considered a conflict of interest at PepsiCo?",
    "What is the policy on accepting gifts from suppliers?",
    "How does PepsiCo handle bribery and international anti-corruption laws?",
    "What are the rules for accurate financial reporting?",
    "What should I do if my supervisor asks me to falsify hours?",
    "How does PepsiCo protect employees from retaliation?",
    "What is the policy on dealing with government officials?",
    "How should I handle a situation where a coworker violates the Code?"
]

def load_question(q):
    return q


# ---------------------------------------------------------
# Gradio UI
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
            PepsiCo Code of Conduct Assistant
        </div>
    """)

    chatbot = gr.Chatbot(label="Conversation History")

    with gr.Row():
        dropdown = gr.Dropdown(
            EXAMPLE_QUESTIONS,
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

    ask_button.click(
        answer_question,
        inputs=[user_input, show_sources, chatbot],
        outputs=chatbot
    )


# ---------------------------------------------------------
# Cloud Run Port Logic
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
