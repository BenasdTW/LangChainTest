import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from helpers import get_collection, transform_retrieval

os.environ["OLLAMA_HOST"] = "http://172.17.0.3:11434"

embedding_model = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5")
llm_model = OllamaLLM(model="llama3.2:1b")

db_path = "./chroma_test"
collection = "collection0"
db = get_collection(collection, embedding_model, db_path)

retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

answer_prompt_template = PromptTemplate(
    input_variables=["question", "formatted_retrieval"],
    template="Answer the following question based on the provided context. Question: {question} Context: {formatted_retrieval}"
)

# llm_chain = RunnableParallel({"question": RunnablePassthrough(), "formatted_retrieval": retriever | RunnableLambda(transform_retrieval2)}) | answer_prompt_template | llm_model
llm_chain = (
    {
        "question": RunnablePassthrough(), 
        "formatted_retrieval":
            retriever
            | RunnableLambda(transform_retrieval)
    }
    | answer_prompt_template
    | llm_model
    | StrOutputParser()
)

question = "What is Isaac Sim"
out = llm_chain.invoke(question)
print(out)
llm_chain.get_graph().print_ascii()
