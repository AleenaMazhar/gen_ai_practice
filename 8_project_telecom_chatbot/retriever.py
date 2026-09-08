from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def build_retriever(
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> RunnableLambda:
    """
    Build a retriever that searches across multiple collections in the Chroma vector store.
    Returns a RunnableLambda that takes a query string and returns a list of relevant documents.
    """

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    faq_store = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name="faq",
        embedding_function=embeddings
    )

    tickets_store = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name="faq",
        embedding_function=embeddings
    )

    guides_store = Chroma(
        persist_directory=CHROMA_DIR,
        collection_name="guides",
        embedding_function=embeddings
    )

    faq_docs = faq_store.as_retriever(search_kwargs={"k": k_faq})
    ticket_docs = tickets_store.as_retriever(search_kwargs={"k": k_tickets})
    guide_docs = guides_store.as_retriever(search_kwargs={"k": k_guides})

    def retrieve(query: str) -> list[Document]:
        return (
            faq_docs.invoke(query)
            + ticket_docs.invoke(query)
            + guide_docs.invoke(query)
        )

    return RunnableLambda(retrieve)
