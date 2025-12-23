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
# LangChain + FAISS RAG Pipeline
# ---------------------------------------------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "PBG_English_3_28.pdf"   # Must be included in the container

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = splitter.split_documents(documents)

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Build FAISS vectorstore
vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.2
)

# Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type="stuff"
)

# Fallback text
FALLBACK_TEXT = """
This question may not be explicitly covered in the Pepsi Bottling Group Worldwide Code of Conduct.

However, the Code emphasizes ethical behavior, legal compliance, anti-bribery rules,
conflict-of-interest prevention, accurate reporting, workplace respect, safety,
and integrity in all business dealings.
"""

# ---------------------------------------------------------
# RAG Answer Function
# ---------------------------------------------------------
def answer_question(question: str) -> str:
    if not question or question.strip() == "":
        return "Please enter a valid question."

    try:
        response = qa_chain.invoke({"query": question})
        answer = response.get("result", "").strip()
        return answer if answer else FALLBACK_TEXT
    except Exception as e:
        return f"Error: {str(e)}"


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

    gr.HTML(f"""
        <div class="pepsico-header">
            PepsiCo Code of Conduct Assistant
        </div>
    """)

    gr.Markdown(
        """
        This assistant helps you explore the Pepsi Bottling Group Worldwide Code of Conduct.
        Select a question or type your own to get a grounded, policy‑aligned answer.
        """
    )

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

    output = gr.Textbox(
        label="Answer",
        lines=12
    )

    gr.Button(
        "Ask",
        variant="primary"
    ).click(answer_question, user_input, output)


# ---------------------------------------------------------
# Cloud Run Port Logic
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
