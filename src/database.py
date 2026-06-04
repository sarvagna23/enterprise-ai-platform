from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./enterprise_ai.db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class AgentRun(Base):
    __tablename__ = "agent_runs"
    id = Column(String, primary_key=True)
    task = Column(Text)
    status = Column(String)
    iterations = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

class RAGQuery(Base):
    __tablename__ = "rag_queries"
    id = Column(String, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def log_agent_run(run_id: str, task: str, status: str, iterations: int):
    session = SessionLocal()
    try:
        run = AgentRun(id=run_id, task=task, status=status, iterations=iterations)
        session.add(run)
        session.commit()
    finally:
        session.close()

def log_rag_query(query_id: str, question: str, answer: str):
    session = SessionLocal()
    try:
        query = RAGQuery(id=query_id, question=question, answer=answer)
        session.add(query)
        session.commit()
    finally:
        session.close()