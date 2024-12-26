import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.prompts import PromptTemplate

os.environ["OLLAMA_HOST"] = "http://172.17.0.3:11434"

llm_model = OllamaLLM(model="llama3.2:3b")
text2sql_model = OllamaLLM(model="duckdb-nsql")

text2sql_prompt = """Here is the database schema that the SQL query will run on:

{table_info}

Natural language query: {question}"""


llm_prompt = """Original user question: {question}

Derived SQL query: {sql_query}

With query result: {query_result}"""


text2sql_prompt_template = PromptTemplate(
    input_variables=["question"],
    template=text2sql_prompt
)
llm_prompt_template = PromptTemplate(
    input_variables=["question", "sql_query", "query_result"],
    template=llm_prompt
)
db = SQLDatabase.from_uri("sqlite:///db/test.db")

text2sql_chain = (
    text2sql_prompt_template
    | text2sql_model
    | RunnableLambda(lambda x: x.strip())
)

llm_chain = (
    llm_prompt_template
    | llm_model
    | RunnableLambda(lambda x: x.strip())
)

question = "Which product has the highest count of test_code_num = 1?"

# Extract the "CREATE TABLE" part
table_info = db.get_table_info(["test_data"])
create_table_info = table_info.split("/*")[0].strip()

sql_query = text2sql_chain.invoke({"question": question, "table_info": create_table_info})
query_result = db.run(sql_query)
out = llm_chain.invoke({"question": question, "sql_query": sql_query, "query_result": query_result})

print(out)

