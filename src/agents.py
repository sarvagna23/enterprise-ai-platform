from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

class AgentState(TypedDict):
    task: str
    plan: Optional[str]
    result: Optional[str]
    review: Optional[str]
    status: str
    iterations: int

def planner_agent(state: AgentState) -> AgentState:
    print(f"[Planner] Planning task...")
    messages = [
        SystemMessage(content="You are a task planner. Break down the given task into clear execution steps."),
        HumanMessage(content=f"Task: {state['task']}\n\nCreate a step-by-step execution plan.")
    ]
    response = llm.invoke(messages)
    return {**state, "plan": response.content, "status": "planned"}

def executor_agent(state: AgentState) -> AgentState:
    print(f"[Executor] Executing plan...")
    messages = [
        SystemMessage(content="You are a task executor. Execute the given plan and provide detailed results."),
        HumanMessage(content=f"Task: {state['task']}\n\nPlan:\n{state['plan']}\n\nExecute this plan and provide results.")
    ]
    response = llm.invoke(messages)
    return {**state, "result": response.content, "status": "executed"}

def monitor_agent(state: AgentState) -> AgentState:
    print(f"[Monitor] Reviewing results...")
    messages = [
        SystemMessage(content="You are a quality monitor. Review task results and determine if they meet requirements."),
        HumanMessage(content=f"Task: {state['task']}\n\nResult:\n{state['result']}\n\nReview the result. Reply with APPROVED or RETRY with reason.")
    ]
    response = llm.invoke(messages)
    review = response.content
    status = "approved" if "APPROVED" in review.upper() else "retry"
    return {**state, "review": review, "status": status, "iterations": state["iterations"] + 1}

# --- ReAct Agent ---
def react_agent(state: AgentState) -> AgentState:
    print(f"[ReAct] Reasoning and acting...")
    messages = [
        SystemMessage(content="""You are a ReAct agent. Follow this pattern:
Thought: Think about what needs to be done
Action: Take a specific action
Observation: What you observe from the action
Repeat until you reach a final answer."""),
        HumanMessage(content=f"Task: {state['task']}\n\nUse ReAct reasoning to solve this step by step.")
    ]
    response = llm.invoke(messages)
    return {**state, "result": response.content, "status": "react_complete"}

def should_retry(state: AgentState) -> str:
    if state["status"] == "retry" and state["iterations"] < 2:
        return "executor"
    return END

def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_agent)
    graph.add_node("executor", executor_agent)
    graph.add_node("monitor", monitor_agent)
    graph.add_node("react", react_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "react")
    graph.add_edge("react", "executor")
    graph.add_edge("executor", "monitor")
    graph.add_conditional_edges("monitor", should_retry, {
        "executor": "executor",
        END: END
    })
    return graph.compile()

if __name__ == "__main__":
    print("Starting agents...")
    graph = build_agent_graph()
    print("Graph built...")
    result = graph.invoke({
        "task": "Analyze the performance of a machine learning model and suggest improvements",
        "plan": None,
        "result": None,
        "review": None,
        "status": "pending",
        "iterations": 0
    })
    print("\n--- Final Result ---")
    print(f"Status: {result['status']}")
    print(f"Plan: {result['plan'][:200]}")
    print(f"Result: {result['result'][:200]}")