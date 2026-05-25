from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from src.agents.extractor import ExtractorAgent
from src.agents.auditor import AuditorAgent

class GraphState(TypedDict):
    query: str
    context: str
    draft: str
    is_valid: bool
    feedback: str
    final_answer: str
    attempts: int

def extractor_node(state: GraphState):
    agent = ExtractorAgent()
    prompt_query = state['query']
    if state.get('feedback'):
        prompt_query += f"\n\nPrevious draft was incorrect. Feedback: {state['feedback']}. Fix the amounts."
    
    draft = agent.generate_draft(prompt_query, state['context'])
    return {"draft": draft, "attempts": state.get('attempts', 0) + 1}

def auditor_node(state: GraphState):
    agent = AuditorAgent()
    print(f"Auditor: Verifying draft...")
    is_valid, feedback = agent.verify(state['draft'], state['context'])
    print(f"Auditor Result: Valid={is_valid}, Feedback={feedback}")
    return {"is_valid": is_valid, "feedback": feedback, "final_answer": state['draft'] if is_valid else ""}

def should_continue(state: GraphState):
    if state.get('is_valid', False) or state.get('attempts', 0) >= 2:
        return "end"
    return "extractor"

def create_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("auditor", auditor_node)
    
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "auditor")
    workflow.add_conditional_edges("auditor", should_continue, {"extractor": "extractor", "end": END})
    
    return workflow.compile()

def run_financial_rag(query, context):
    graph = create_graph()
    initial_state = {"query": query, "context": context, "attempts": 0}
    result = graph.invoke(initial_state)
    return result.get('final_answer') or result.get('draft')
