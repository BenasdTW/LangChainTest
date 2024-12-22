import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["OLLAMA_HOST"] = "http://172.17.0.3:11434"

embedding_model = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5")
llm_model = OllamaLLM(model="llama3.2:1b")

pdf_path = "./test.pdf"

loader = PyPDFLoader(pdf_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=50)

texts = loader.load_and_split(splitter)
db = Chroma.from_documents(texts, embedding_model, persist_directory="./chroma_test")

query = "Isaac Sim"
docs = db.similarity_search(query, k=3)

for doc in docs:
    print(doc.page_content)
    print("=" * 60)

# for i in loader.load_and_split(splitter):
#     print(i.page_content)
#     print("=" * 60)