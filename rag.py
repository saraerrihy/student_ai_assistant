# rag.py
import os
import pickle
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

INDEX_FILE = "storage.pkl"
CHROMA_DIR = Path("./chroma_db")

def build_index(data_folder: str):
    txt_files = list(Path(data_folder).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Aucun fichier .txt trouvé dans {data_folder}")

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()) and os.path.exists(INDEX_FILE):
        embeddings = OllamaEmbeddings(model="llama3.2:3b")
        vectorstore = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
        print("✅ Base vectorielle chargée depuis le disque.")
        return vectorstore

    print("🔨 Construction de la base vectorielle à partir des fichiers TXT...")
    documents = []
    for txt in txt_files:
        try:
            loader = TextLoader(str(txt), encoding="utf-8")
            documents.extend(loader.load())
            print(f"   - TXT chargé : {txt.name}")
        except Exception as e:
            print(f"   ⚠️ Erreur lecture {txt.name}: {e}")

    if not documents:
        raise RuntimeError("Aucun document n'a pu être chargé.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    vectorstore.persist()
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"indexed": True}, f)

    print(f"✅ Base vectorielle créée avec {len(chunks)} chunks.")
    return vectorstore

def get_retriever(data_folder: str = "./data"):
    vectorstore = build_index(data_folder)
    return vectorstore.as_retriever(search_kwargs={"k": 5})