import os
from langchain_ollama import OllamaEmbeddings
from helpers import create_basic_pdf_rag, db_formatted_output

os.environ["OLLAMA_HOST"] = "http://172.17.0.3:11434"

embedding_model = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5")

pdf_path = "./test.pdf"
db_path = "./chroma_test"
collection = "collection0"

db = create_basic_pdf_rag(
    pdf_path,
    chunk_size=500,
    chunk_overlap=100,
    collection=collection,
    embedding_model=embedding_model,
    db_path=db_path,
)

query = "Isaac Sim"
print(db_formatted_output(db, query, k=3))

# for i in loader.load_and_split(splitter):
#     print(i.page_content)
#     print("=" * 60)