import chromadb
from sentence_transformers import SentenceTransformer
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

# --- Vector DB Setup ---
client = chromadb.Client()
collection = client.create_collection("enterprise_docs")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# --- Index Documents ---
def index_documents(docs: list[dict]):
    texts = [d["text"] for d in docs]
    ids = [d["id"] for d in docs]
    embeddings = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids)
    print(f"Indexed {len(docs)} documents")

# --- Retrieve ---
def retrieve(query: str, top_k: int = 3) -> list[str]:
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)
    return results["documents"][0]

# --- RAG Query ---
def rag_query(question: str) -> str:
    context_docs = retrieve(question)
    context = "\n\n".join(context_docs)
    messages = [
        SystemMessage(content="You are an enterprise AI assistant. Answer questions based on the provided context."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}")
    ]
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    # Sample enterprise docs
    docs = [
        {"id": "1", "text": "Our fraud detection system uses XGBoost achieving 99.84% accuracy on 284K transactions."},
        {"id": "2", "text": "The federated learning framework coordinates 10 nodes with Byzantine fault detection achieving 98.9% reliability."},
        {"id": "3", "text": "CI/CD pipeline reduces deployment time by 45% using Docker and Terraform on AWS."},
        {"id": "4", "text": "The ML platform tracks experiments, model versions, and performance metrics in real time."},
    ]
    index_documents(docs)
    answer = rag_query("What is the accuracy of the fraud detection system?")
    print(f"\nRAG Answer: {answer}")