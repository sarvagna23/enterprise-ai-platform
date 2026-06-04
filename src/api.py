from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import sys
import os
import uuid
sys.path.insert(0, os.path.dirname(__file__))

from agents import build_agent_graph
from rag import index_documents, rag_query
from database import init_db, log_agent_run, log_rag_query

app = FastAPI(title="Enterprise AI Agent Platform", version="1.0.0")

graph = build_agent_graph()

sample_docs = [
    {"id": "1", "text": "Our fraud detection system uses XGBoost achieving 99.84% accuracy on 284K transactions."},
    {"id": "2", "text": "The federated learning framework coordinates 10 nodes with Byzantine fault detection achieving 98.9% reliability."},
    {"id": "3", "text": "CI/CD pipeline reduces deployment time by 45% using Docker and Terraform on AWS."},
    {"id": "4", "text": "The ML platform tracks experiments, model versions, and performance metrics in real time."},
]
index_documents(sample_docs)
init_db()

class TaskRequest(BaseModel):
    task: str

class QueryRequest(BaseModel):
    question: str

class TaskResponse(BaseModel):
    run_id: str
    status: str
    plan: str
    result: str
    review: Optional[str]
    iterations: int

class QueryResponse(BaseModel):
    query_id: str
    answer: str

@app.get("/health")
def health():
    return {"status": "ok", "service": "Enterprise AI Agent Platform"}

@app.post("/agent/run", response_model=TaskResponse)
def run_agent(request: TaskRequest):
    try:
        result = graph.invoke({
            "task": request.task,
            "plan": None,
            "result": None,
            "review": None,
            "status": "pending",
            "iterations": 0
        })
        run_id = str(uuid.uuid4())
        log_agent_run(run_id, request.task, result["status"], result["iterations"])
        return TaskResponse(
            run_id=run_id,
            status=result["status"],
            plan=result["plan"],
            result=result["result"],
            review=result.get("review"),
            iterations=result["iterations"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        answer = rag_query(request.question)
        query_id = str(uuid.uuid4())
        log_rag_query(query_id, request.question, answer)
        return QueryResponse(query_id=query_id, answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/react")
def run_react_agent(request: TaskRequest):
    try:
        result = graph.invoke({
            "task": request.task,
            "plan": None,
            "result": None,
            "review": None,
            "status": "pending",
            "iterations": 0
        })
        return {
            "status": result["status"],
            "reasoning": result["result"],
            "iterations": result["iterations"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)