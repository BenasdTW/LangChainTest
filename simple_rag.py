import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.chains.transform import TransformChain
from helpers import get_collection, transform_retrieval

os.environ["OLLAMA_HOST"] = "http://172.17.0.3:11434"

embedding_model = OllamaEmbeddings(model="quentinz/bge-large-zh-v1.5")
llm_model = OllamaLLM(model="llama3.2:1b")

db_path = "./chroma_test"
collection = "collection0"
db = get_collection(collection, embedding_model, db_path)

# Define the prompt template for answering the question
answer_prompt_template = PromptTemplate(
    input_variables=["question", "formatted_retrieval"],
    template="Answer the following question based on the provided context. Question: {question} Context: {formatted_retrieval}"
)
question = "What is Isaac Sim"
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# Wrap the retriever in a TransformChain to provide input_keys and output_keys
retriever_chain = TransformChain(
    input_variables=["question"],
    output_variables=["docs"],
    transform=lambda inputs: {"docs": retriever.invoke(inputs["question"])}
)

transform_chain = TransformChain(
    input_variables=["docs"],
    output_variables=["formatted_retrieval"],
    transform=transform_retrieval,
)
llm_chain = LLMChain(llm=llm_model, prompt=answer_prompt_template, output_key="answer")

sequential_chain = SequentialChain(
    chains=[retriever_chain, transform_chain, llm_chain],
    input_variables=["question"],
    output_variables=["answer"]
)
out = sequential_chain.invoke(question)
print(out)
sequential_chain.get_graph().print_ascii()
