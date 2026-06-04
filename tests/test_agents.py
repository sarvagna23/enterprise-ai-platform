import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_rag_query():
    response = client.post("/rag/query", json={"question": "What is the fraud detection accuracy?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "query_id" in data
    assert len(data["answer"]) > 0

def test_rag_query_federated():
    response = client.post("/rag/query", json={"question": "Tell me about federated learning"})
    assert response.status_code == 200
    assert "answer" in response.json()

def test_health_service_name():
    response = client.get("/health")
    assert "service" in response.json()