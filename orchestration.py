# orchestration.py
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from agents import (
    chercheur_agent, expliqueur_agent,
    resumeur_agent, verificateur_agent
)

class AssistantState(TypedDict):
    query: str
    raw_infos: str
    explanation: str
    summary: str
    verification: str
    iteration: int

def run_agent(agent, input_text: str) -> str:
    result = agent.invoke({"messages": [("user", input_text)]})
    return result["messages"][-1].content

def chercheur_node(state: AssistantState) -> dict:
    print("🔍 [Chercheur] Interrogation du document...")
    return {"raw_infos": run_agent(chercheur_agent, state["query"])}

def expliqueur_node(state: AssistantState) -> dict:
    print("📖 [Expliqueur] Rédaction de l'explication...")
    prompt = f"""Voici les informations brutes extraites du document :
{state['raw_infos']}

Question : {state['query']}

Produis une explication claire."""
    return {"explanation": run_agent(expliqueur_agent, prompt)}

def resumeur_node(state: AssistantState) -> dict:
    print("📝 [Résumeur] Synthèse...")
    prompt = f"Résume en 2-3 phrases : {state['explanation']}"
    return {"summary": run_agent(resumeur_agent, prompt)}

def verificateur_node(state: AssistantState) -> dict:
    print("✅ [Vérificateur] Vérification...")
    prompt = f"""Document source : contenu du cours.
Explication à vérifier : {state['explanation']}
Question initiale : {state['query']}

Applique les règles : réponds uniquement VALIDÉ ou À REVOIR."""
    return {"verification": run_agent(verificateur_agent, prompt)}

# Construction du graphe
builder = StateGraph(AssistantState)
builder.add_node("chercheur", chercheur_node)
builder.add_node("expliqueur", expliqueur_node)
builder.add_node("resumeur", resumeur_node)
builder.add_node("verificateur", verificateur_node)

builder.add_edge(START, "chercheur")
builder.add_edge("chercheur", "expliqueur")
builder.add_edge("expliqueur", "resumeur")
builder.add_edge("resumeur", "verificateur")
builder.add_edge("verificateur", END)

app = builder.compile()