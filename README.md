# Projet SMA - Assistant Étudiant Multi-Agents

## Contexte
Projet réalisé dans le cadre du module **IA Distribuée & Systèmes Multi-Agents**.
L'objectif est de construire un système multi-agents intelligent utilisant le RAG (Retrieval-Augmented Generation) et une orchestration avancée.

## Architecture
![Schéma](https://via.placeholder.com/600x300?text=Schema+archi+%C3%A0+compl%C3%A9ter)

4 agents collaborant via un graphe d'état LangGraph :
1. **Chercheur** : interroge la base de connaissances (RAG)
2. **Expliqueur** : reformule pédagogiquement
3. **Résumeur** : synthétise en puces
4. **Vérificateur** : valide ou demande révision

## Pipeline RAG (LlamaIndex)
- **Ingestion** : fichiers `.txt` dans `data/`
- **Chunking** : découpage par défaut (512 tokens, overlapping 20%)
- **Embeddings** : `BAAI/bge-small-en` (local)
- **Vector store** : ChromaDB (persistant)
- **Retrieval** : `similarity_top_k=3`

## Agents & Outils (LangChain)
Chaque agent dispose d'un outil `get_current_time` (date/heure).  
Le Chercheur utilise en plus un outil de recherche RAG.

## Orchestration (LangGraph)
État partagé contenant `query`, `raw_infos`, `explanation`, `summary`, `verification`, `iteration`.  
Flux : `START → chercheur → expliqueur → résumeur → vérificateur → (VALIDÉ → END | REVISE → expliqueur)`.

## Technologies
- **LLM** : Groq (modèle `llama-3.3-70b-versatile`) – gratuit et rapide
- **Framework agents** : LangChain
- **RAG** : LlamaIndex + ChromaDB
- **Orchestration** : LangGraph

## Installation

```bash
git clone ...
cd student_ai_assistant
pip install -r requirements.txt