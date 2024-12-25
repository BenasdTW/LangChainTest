import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from helpers import get_collection, transform_retrieval

os.environ["OLLAMA_HOST"] = "http://172.17.0.3:11434"

embedding_model = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5")
llm_model = OllamaLLM(model="llama3.2:3b")

prompt = PromptTemplate.from_template(
    """What is the language of the target text below. Answer either `English`, `Chinese`.

Do not respond with more than one word.

<target>
{question}
</target>

Classification:"""
)

eng_prompt = PromptTemplate(
    input_variables=["question", "formatted_retrieval"],
    template="Answer the following question based on the provided context. Question: {question} Context: {formatted_retrieval}"
)

zh_prompt = PromptTemplate(
    input_variables=["question", "formatted_retrieval"],
    template="根據提供的資料，以繁體中文回答問題。 問題： {question} 資料： {formatted_retrieval}"
)

db_path = "./chroma_test"

db0 = get_collection("collection0", embedding_model, db_path)
db1 = get_collection("collection1", embedding_model, db_path)

eng_retriever = db0.as_retriever(
    search_kwargs={"k": 3}
)
zh_retriever = db1.as_retriever(
    search_kwargs={"k": 3}
)

eng_chain = (
    {
        "question": RunnablePassthrough(),
        "formatted_retrieval":
            RunnableLambda(lambda x: x["question"])
            | eng_retriever
            | RunnableLambda(transform_retrieval)
    }
    | eng_prompt
)
zh_chain = (
    {
        "question": RunnablePassthrough(),
        "formatted_retrieval":
            RunnableLambda(lambda x: x["question"])
            | zh_retriever
            | RunnableLambda(transform_retrieval)
    }
    | zh_prompt
)

def route(info):
    if "English" in info["lang"]:
        return eng_chain
    else:
        return zh_chain

llm_chain = (
    {
        "question": RunnablePassthrough(),
        "lang": prompt | llm_model
    }
    | RunnableLambda(route)
    | llm_model
    | StrOutputParser()
)

# print(chain.invoke({"question": "What is Isaac Sim? Answer in English."}))
# print(chain.invoke({"question": "什麼是 Isaac Sim 模擬器? 用繁體中文回答。"}))
# print(llm_chain.invoke("What is Isaac Sim? Answer in English."))
print(llm_chain.invoke("什麼是 Isaac Sim 模擬器? 用繁體中文回答。"))

