import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

_vector_db = None

def _get_vector_db():
    global _vector_db
    if _vector_db is not None:
        return _vector_db

    knowledge_path = os.path.join("knowledge", "tumor_facts.txt")
    if not os.path.exists(knowledge_path):
        os.makedirs("knowledge", exist_ok=True)
        with open(knowledge_path, "w") as f:
            f.write("""
            Glioma: A type of tumor that starts in the glial cells of the brain or spine.
            Meningioma: A tumor that arises from the meninges—the membranes that surround your brain and spinal cord. Most meningiomas are noncancerous (benign).
            Pituitary Tumor: Abnormal growths that develop in your pituitary gland. Most pituitary tumors are noncancerous (benign) adenomas.
            No Tumor: The brain MRI scan exhibits clear standard physiological tissue structures without identifiable abnormal masses.
            """)

    loader = TextLoader(knowledge_path)
    documents = loader.load()
    ttext_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    _vector_db = FAISS.from_documents(docs, embeddings)
    return _vector_db

def query_knowledge_base(query: str) -> str:
    try:
        db = _get_vector_db()
        results = db.similarity_search(query, k=2)
        if not results:
            return "No specific reference documents matched the query."
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Knowledge retrieval note: {str(e)}"
