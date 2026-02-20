"""
Health Insurance Policy RAG using LangGraph
-------------------------------------------
- PDF ingestion
- Vector search (Chroma)
- Grounded LLM generation
- Verification node
- LangSmith tracing enabled

Run:
    python health_policy_rag.py

prerequisites:
pip install langchain langgraph langsmith \
langchain_google_genai langchain-community \
pypdf chromadb tiktoken

"""

import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
# Load .env file
load_dotenv()
# Get the API key
api_key = os.getenv("GOOGLE_API_KEY")

# =========================
#  ENV CONFIG
# =========================

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG-Health-Insurance"

# Set your keys before running:
# export LANGCHAIN_API_KEY="your-langsmith-key"


# =========================
#  BUILD VECTOR STORE
# =========================

def build_vector_store(pdf_paths: List[str]):

    documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)

        docs = loader.load()
        print(f"Loaded {len(docs)} pages from {path}")
        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)

    # list the models in GenAI
    """genai.configure(api_key=api_key)

    for m in genai.list_models():
        print(m.name, m.supported_generation_methods)"""

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )   

    vectordb = Chroma.from_documents(
        split_docs,
        embedding=embeddings,
        persist_directory="./policy_db"
    )

    return vectordb


# =========================
#  STATE DEFINITION
# =========================

class PolicyState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    answer: str


# =========================
# 🔎 RETRIEVER NODE
# =========================

def retrieve_node(state: PolicyState):

    docs = vectordb.similarity_search(state["question"], k=4)

    return {"retrieved_docs": docs}


# =========================
#  GENERATION NODE
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,# deterministic
    google_api_key=api_key
)

def generate_answer_node(state: PolicyState):

    context = "\n\n".join(
    [
        f"""
        Source {i+1}
        File: {doc.metadata.get('source')}
        Page: {doc.metadata.get('page')}
        Content:
        {doc.page_content}
        """
                for i, doc in enumerate(state["retrieved_docs"])
            ]
        )


    prompt = ChatPromptTemplate.from_template("""
    You are an expert health insurance assistant.

    Rules:
    - Answer ONLY from provided context.
    - Cite using format:
    (File: <filename>, Page: <page number>)
    - If not found, say:
    "I could not find this in the policy document."

    Context:
    {context}

    Question:
    {question}
    """)


    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": state["question"]
    })

    return {"answer": response.content}


# =========================
#  VERIFICATION NODE
# =========================

def verify_node(state: PolicyState):

    verification_prompt = f"""
    You are verifying if the answer is fully supported by the context.

    Context:
    {[doc.page_content for doc in state['retrieved_docs']]}

    Answer:
    {state['answer']}

    Respond ONLY with YES or NO.
    """

    verdict = llm.invoke(verification_prompt)

    if "NO" in verdict.content.upper():
        return {
            "answer": "The answer could not be verified from the policy document."
        }

    return {}


# =========================
#  BUILD GRAPH
# =========================

def build_graph():

    graph = StateGraph(PolicyState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


# =========================
#  MAIN EXECUTION
# =========================

if __name__ == "__main__":

    # Provide your policy PDFs here
    pdf_files = [
        "./policy_docs/ultimate-care-policy-tnc-january-2026.pdf",
    ]

    global vectordb
    vectordb = build_vector_store(pdf_files)

    rag_app = build_graph()

    print("\n🏥 Health Insurance Policy Assistant")
    print("Type 'exit' to quit.\n")

    while True:

        user_question = input("Ask a question: ")

        if user_question.lower() == "exit":
            break

        result = rag_app.invoke({
            "question": user_question
        })

        print("\n📘 Answer:")
        print(result["answer"])
        print("\n" + "-"*50 + "\n")
