import os
os.environ["TRANSFOERMERS_VERBOSITY"] = "error"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CROMA_DIR = "chroma_store"
COLLECTION = "guides"
PDF_PATH = os.path.join("data", "telecom_guide.pdf")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def main():
    print("Loading PDF document...")
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages. Splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", "."]
    )

    chunks = splitter.split_documents(pages)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = i

    print(f"Created {len(chunks)} chunks. Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CROMA_DIR,
        collection_name=COLLECTION
    )

    print("Persisting vectorstore to disk...")

if __name__ == "__main__":
    main()
