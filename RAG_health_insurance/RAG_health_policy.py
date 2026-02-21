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
    chat_history: list
    retrieved_docs: List[Document]
    answer: str


# =========================
# Intent classification NODE
# =========================

def classify_question_node(state: PolicyState):
    # check if the new question is a followup or new topic.
    # this is to handle context bleed across topic shifts

    history_text = ""
    for turn in state["chat_history"]:
        history_text += f"{turn['role']}: {turn['content']}\n"

    prompt = f"""
    You are part of a Q&A RAG based on a database. 
    All questions would be related to the database.
    Your job is to check if the current question is a continuation to the previous (history).

    for example, if previous and current queston are related to "waiting period", treat them as relevant.
    If previous question is about "waiting period" and current question is about "exclusions", dont treat them as relevant though the are from same policy.
    Conversation so far:
    {history_text}

    New Question:
    {state["question"]}

    Is the new question:
    A) A follow-up to the previous sub-topic
    B) A new and unrelated sub-topic

    Respond ONLY with A or B.
    """

    verdict = llm.invoke(prompt).content.strip()
    print("classify_question_node Verdict: ",verdict)

    if verdict == "B":
        print("Reset History....")
        # reset history
        return {"chat_history": []}

    return {}

# =========================
# REWRITE QUESTION NODE
# =========================
def rewrite_question_node(state):

    history_text = ""
    for turn in state["chat_history"]:
        history_text += f"{turn['role']}: {turn['content']}\n"

    prompt = f"""
    Rewrite the user's latest question into a standalone question.
    Preserve meaning but remove ambiguity.

    Conversation:
    {history_text}

    Question:
    {state['question']}
    """

    rewritten = llm.invoke(prompt).content

    print("rewritten question: ", rewritten)

    return {"question": rewritten}


# =========================
# RETRIEVER NODE
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

    # Format chat history
    history_text = ""
    for turn in state["chat_history"]:
        history_text += f"{turn['role']}: {turn['content']}\n"

    prompt = ChatPromptTemplate.from_template("""
    You are an expert health insurance assistant.

    Conversation so far:
    {chat_history}          
                                                                             
    Rules:
    - Answer ONLY from provided context.
    - if chat_history is present, the current question is a followup from previous. So consider it begore answering.                                
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
        "chat_history": history_text,
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
#  CLASSIFIER ROUTER
# =========================
def classifier_router(state: PolicyState):

    if state["chat_history"]:
        #return "publish"
        return "rewrite"

    return "retrieve"


# =========================
#  BUILD GRAPH
# =========================

def build_graph():

    graph = StateGraph(PolicyState)

    graph.add_node("classify", classify_question_node)
    graph.add_node("rewrite",rewrite_question_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
    "classify",
    classifier_router,
    {
        "rewrite": "rewrite",
        "retrieve": "retrieve"
    }
    )

    graph.add_edge("rewrite", "retrieve")
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
    #vectordb = build_vector_store(pdf_files)
    # check if the db already exists. If exits, load the existing db, else create one
    if os.path.exists("./policy_db"):
        print("Loading existing vector DB...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        vectordb = Chroma(
            persist_directory="./policy_db",
            embedding_function=embeddings
        )
    else:
        print("Building new vector DB...")
        vectordb = build_vector_store(pdf_files)

    rag_app = build_graph()

    print("\n Health Insurance Policy Assistant")
    print("Type 'exit' to quit.\n")

    chat_history = []

    while True:

        user_question = input("Ask a question: ")

        if user_question.lower() == "exit":
            break

        result = rag_app.invoke({
            "question": user_question,
            "chat_history": chat_history
        })

        print("\n📘 Answer:")
        print(result["answer"])
        print("\n" + "-"*50 + "\n")

        # Update memory
        chat_history.append({"role": "user", "content": user_question})
        chat_history.append({"role": "assistant", "content": result["answer"]})
