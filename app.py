# app.py
import streamlit as st
import os
from rag import build_index
from orchestration import app

@st.cache_resource
def init_index():
    if not os.path.exists("storage.pkl"):
        build_index("./data")
    else:
        build_index("./data")
    return app

st.set_page_config(page_title="Assistant Multi-Agents", layout="wide")
st.title("🤖 Assistant Étudiant - Système Multi-Agents")
st.markdown("Pose une question sur le cours. Les agents Chercheur → Expliqueur → Résumeur → Vérificateur collaborent pour te répondre.")

orchestrator = init_index()

question = st.text_input("🧑‍🎓 Ta question :", placeholder="Ex: Qu'est-ce que l'architecture ReAct ?")

if st.button("Envoyer", type="primary") and question:
    with st.spinner("Les agents travaillent..."):
        result = orchestrator.invoke({"query": question, "iteration": 0})
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🔍 Chercheur (extraits bruts)", expanded=True):
            st.write(result["raw_infos"])
        with st.expander("📖 Expliqueur (explication détaillée)"):
            st.write(result["explanation"])
    with col2:
        with st.expander("📝 Résumeur (synthèse)"):
            st.write(result["summary"])
        with st.expander("✅ Vérificateur (contrôle qualité)"):
            verdict = result["verification"]
            if "VALIDÉ" in verdict.upper():
                st.success(verdict)
            else:
                st.warning(verdict)
    st.success("Réponse terminée.")

st.markdown("---")
st.caption("Système multi-agents local avec Ollama + LangGraph | Cours intégré dans data/cours.txt")
