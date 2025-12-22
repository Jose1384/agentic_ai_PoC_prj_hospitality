import asyncio
from entities.embeddings import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
import os
import uuid
from datetime import datetime, timezone
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    JSONLoader,
    TextLoader,
)
from config.agent_config import _get_env_value, _load_config_file
from util.configuration import PROJECT_ROOT
from langchain_chroma import Chroma
from db.session_sqlalchemy import DatabaseSessionManager, transaction_db_async

HOTELS_DATA_PATH_LOCAL = PROJECT_ROOT / "data" / "hotels"
HOTELS_DATA_PATH_EXTERNAL = PROJECT_ROOT.parent / "bookings-db" / "output_files" / "hotels"
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma" / "hotels"


# -----------------------------
# Initialize cache embeddings
#
# Info: Store persists embeddings to the local filesystem
# This isn't for production use, but is useful for local
# -----------------------------
def _initialize_cache_embeddings():    
    embeddings  = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key= _get_env_value("AI_AGENTIC_API_KEY"))
    store = LocalFileStore("cache/")
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        store,
        namespace=embeddings.model
    )


# -----------------------------
# Document chunking
# -----------------------------
def _get_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False)


## Logic service to process documents
def process_documents(raw_dir: str=HOTELS_DATA_PATH_LOCAL):
    settings = _load_config_file()
    splitter = _get_text_splitter(
        chunk_size=settings.get("rag", {}).get("chunk_size", 1000),
        chunk_overlap=settings.get("rag", {}).get("chunk_overlap", 200)
    )
    all_docs = []
    for root, _, files in os.walk(raw_dir):
        for file in files:
            file_path = os.path.join(root, file)
            docs = _load_file(file_path)
            if not docs:
                continue
            for doc in docs:
                # metadata base
                doc.metadata["doc_id"] = str(uuid.uuid4())
                doc.metadata["title"] = os.path.splitext(file)[0]
                doc.metadata["source"] = "internal"
                doc.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
            # chunking
            chunks = splitter.split_documents(docs)
            all_docs.extend(chunks)
            print(f"[INFO] {file_path}: {len(chunks)} chunks")
    return all_docs

def _load_file(file_path: str):
    """Load a single file based on its extension."""
    ext = file_path.lower()
    try:
        if ext.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif ext.endswith(".docx"):
             loader = Docx2txtLoader(file_path)
        elif ext.endswith(".csv"):
             loader = CSVLoader(file_path, encoding="utf-8")
        elif ext.endswith(".jsonl"):
             loader = JSONLoader(file_path, jq_schema=".", text_content=False, json_lines=True)
        elif ext.endswith(".json"):
             loader = JSONLoader(file_path, jq_schema=".", text_content=False)
        elif ext.endswith(".md") or ext.endswith(".txt"):
             loader = TextLoader(file_path, encoding="utf-8")
        elif ext.endswith(".txt"):
             loader = TextLoader(file_path, encoding="utf-8")
        else:
             print(f"[SKIP] Formato no soportado: {file_path}")
             return []

        docs = loader.load()

        # Si es JSON/JSONL, convertimos dict/list a texto legible
        if ext.endswith(".json") or ext.endswith(".jsonl"):
              for d in docs:
                  if not isinstance(d.page_content, str):
                      d.page_content = DocumentUtils._json_to_text(d.page_content).strip()

        # Asegurar que page_content sea string
        for d in docs:
            if not isinstance(d.page_content, str):
                try:
                    d.page_content = str(d.page_content)
                except Exception:
                    d.page_content = ""

        return docs
    except Exception as e:
        print(f"[ERROR] Cargando {file_path}: {e}")
        return []


## Logic service to upsert documents(embeddings) in chroma
async def upsert_documents_chroma(docs):
    settings = _load_config_file()
    total = len(docs)
    batch_size = settings.get("rag", {}).get("batch_size", 10)

    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]

        # Store embeddings
        try:
            #embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            #GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key= _get_env_value("AI_AGENTIC_API_KEY"))
            embeddings  = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key= _get_env_value("AI_AGENTIC_API_KEY"))
            Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory="../vector_store/chroma_db"
            )
        except Exception as e:
            print(f"[ERROR] Generating batch embeddings {i // batch_size + 1}: {e}")
            raise

# insert embeddings in pgvector(not tested) 



# db_manager global para inicialización lazy - not in production
db_manager = None
async def upsert_documents_pgvector(docs):
    global db_manager
    settings = _load_config_file()
    total = len(docs)
    batch_size = settings.get("rag", {}).get("batch_size", 10)

    db_uri_async = _get_env_value("DB_URI_ASYNC")
    if not db_uri_async:
        raise RuntimeError("DB_URI_ASYNC was not found in environment variables.")

    if db_manager is None:
        db_manager = DatabaseSessionManager(db_uri_async)

    for i in range(0, total, batch_size):
        batch = docs[i:i + batch_size]
        texts = [d.page_content for d in batch]

        try:
            cached_embedder = _initialize_cache_embeddings()
            embs = cached_embedder.embed_documents(texts)
        except Exception as e:
            print(f"[ERROR] Generando embeddings batch {i // batch_size + 1}: {e}")
            raise e

        async with transaction_db_async(db_manager) as db_conn:
            for doc, emb in zip(batch, embs):
                # UUIDv5 estable por chunk basado en contenido y título
                content_for_id = f"{doc.metadata.get('title', '')}-{doc.page_content}"
                stable_uuid = uuid.uuid5(uuid.NAMESPACE_URL, content_for_id)

                created_at_str = doc.metadata.get("created_at")
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)

                embedding_entry = Document(
                    doc_id=str(stable_uuid),
                    title=doc.metadata.get("title", ""),
                    content=doc.page_content,
                    embedding=emb,
                    source=doc.metadata.get("source", ""),
                    date=created_at
                )

                # merge → si existe actualiza, si no insert
                await db_conn.merge(embedding_entry)

        print(f"[INFO] Upserted batch {i //batch_size + 1} ({len(batch)} docs)")


# -----------------------------
# Ingest Pipeline
# -----------------------------
if __name__ == "__main__":
    settings = _load_config_file()
    print("[INFO] Loading and processing documents...")
    docs = process_documents(HOTELS_DATA_PATH_EXTERNAL)
    print(f"[INFO] Total chunks generados: {len(docs)}")
    if docs:
        type_db = settings.get("rag", {}).get("db", {}).get("type", "chroma")
        if type_db == "chroma":
            print("[INFO] Upserting documents into ChromaDB...")
            asyncio.run(upsert_documents_chroma(docs))
            print("[INFO] Processing completed. Documents on ChromaDB")
        elif type_db == "pgvector":
            print("[INFO] Upserting documents into pgvector...")
            asyncio.run(upsert_documents_pgvector(docs))
            print("[INFO] Processing completed. Documents on pgvector")
    else:
        print("[INFO] No documents were generated to upload.")