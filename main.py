# main.py
import os
from rag import build_index
from orchestration import app

def main():
    if not os.path.exists("storage.pkl"):
        print("🔨 Construction initiale de l'index vectoriel...")
        build_index("./data")
    else:
        print("✅ Index existant, chargement...")
        build_index("./data")

    print("\n🤖 Assistant étudiant multi-agents (basé sur cours.txt) prêt.")
    print("Pose tes questions. Tape 'exit' ou 'quit' pour quitter.\n")
    
    while True:
        query = input("🧑‍🎓 Question : ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Au revoir !")
            break
        if not query.strip():
            continue
        
        result = app.invoke({"query": query, "iteration": 0})
        print("\n" + "="*50)
        print("🔍 CHERCHEUR :\n", result["raw_infos"])
        print("\n📖 EXPLIQUEUR :\n", result["explanation"])
        print("\n📝 RÉSUMÉ :\n", result["summary"])
        print("\n✅ VÉRIFICATEUR :\n", result["verification"])
        print("="*50 + "\n")

if __name__ == "__main__":
    main()