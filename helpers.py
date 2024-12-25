import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def list_collections(path):
    client = chromadb.PersistentClient(path=path)
    # List all collections in the database
    collections = client.list_collections()
    collection_names = [collection.name for collection in collections]
    return collection_names

def get_collection(collection, embedding_model, db_path):
    collections = list_collections(db_path)
    # print(f"{collections=}")
    if not collection in collections:
        raise Exception("Collection not found")
    else:
        db = Chroma(collection, embedding_model, persist_directory=db_path)
    return db

def create_basic_pdf_rag(pdf_path, chunk_size, chunk_overlap, collection, embedding_model, db_path):
    loader = PyPDFLoader(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = loader.load_and_split(splitter)

    collections = list_collections(db_path)

    if collection in collections:
        print("Overwriting existing collection")
        db = Chroma(collection, embedding_model, persist_directory=db_path)
        db.delete_collection()

    db = Chroma.from_documents(texts, embedding_model, collection_name=collection, persist_directory=db_path)

    return db

def db_formatted_output(db, query, k=3):
    docs = db.similarity_search(query, k=k)
    output = ""
    for doc in docs:
        output += doc.page_content + "\n" + "=" * 60 + "\n"
        # print(doc.page_content)
        # print("=" * 60)
    return output

def transform_retrieval(docs):
    output = ""
    for doc in docs:
        output += doc.page_content + "\n" + "=" * 60 + "\n"

    return output

# print(f"{list_collections("./chroma_test")=}")
