# agents.py
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from rag import get_retriever

llm = ChatOllama(model="llama3.2:3b", temperature=0)
retriever = get_retriever("./data")

@tool
def recherche_dans_lab_pdf(query: str) -> str:
    """Interroge le fichier texte (data/cours.txt) pour trouver des informations précises."""
    docs = retriever.invoke(query)
    if not docs:
        return "Aucune information trouvée dans le document."
    return "\n\n".join([doc.page_content for doc in docs])

# Agent Chercheur
chercheur_agent = create_react_agent(
    model=llm,
    tools=[recherche_dans_lab_pdf],
    prompt="""Tu es l'agent Chercheur. Utilise l'outil 'recherche_dans_lab_pdf' pour extraire des passages bruts du document. Ne reformule pas, donne des citations."""
)

# Agent Expliqueur
expliqueur_agent = create_react_agent(
    model=llm,
    tools=[recherche_dans_lab_pdf],
    prompt="""Tu es l'agent Expliqueur. Produis une explication claire et pédagogique à partir des informations fournies."""
)

# Agent Résumeur
resumeur_agent = create_react_agent(
    model=llm,
    tools=[],
    prompt="""Tu es l'agent Résumeur. Résume l'explication fournie en 2 ou 3 phrases."""
)

# Agent Vérificateur (corrigé)
verificateur_agent = create_react_agent(
    model=llm,
    tools=[recherche_dans_lab_pdf],
    prompt="""Tu es l'agent Vérificateur. Ta mission est de juger si l'explication fournie par l'agent Expliqueur est correcte, complète et fidèle au document source.

Règles strictes :
- Tu réponds UNIQUEMENT par 'VALIDÉ' ou 'À REVOIR'.
- Si 'À REVOIR', tu ajoutes en une phrase courte ce qui manque ou ce qui est faux.
- Tu ne reformules pas l'explication. Tu ne donnes pas de nouvelle information.

Exemple de réponse attendue :
VALIDÉ
ou
À REVOIR : L'explication oublie de mentionner le rôle du Reviewer.
"""
)