from pyalex import Works

import time, os, re
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chains import RetrievalQA, LLMChain, TransformChain, SequentialChain, ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from langchain import PromptTemplate, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import RegexParser, CommaSeparatedListOutputParser

vector_store = None

csv_list_parser = CommaSeparatedListOutputParser()
expand_query_output_parser = RegexParser(
  regex=r"(?i)((?:.*?\n)*)\nAlternative queries:(.*)",
  output_keys=["reasoning", "queries"],
)

def process_queries_output(inputs: dict) -> dict:
  output_parts = expand_query_output_parser.parse(inputs["expand_query_output"])
  print(output_parts["reasoning"])
  return {"queries": csv_list_parser.parse(output_parts["queries"])}

def create_query_expander():
  expand_query_template = """You are an AI search engine for scientific papers. You are given a query and you must provide complementary search queries, to help the user find the relevant information.
Query may be a question, search for the appropriate content. No more than 5 queries, pick the ones that would yield the most relevant and diverse papers.
If there is any problem with the query, just reply with the original query.

# Example
Query: machine learning rnn
Machine Learning (ML) and Recurring Neural Networks (RNNs). LSTMs, GRU, etc.
Alternative queries: machine learning rnn, lstm, machine learning gru, attention mechanism, machine learning transformers

# Example
Query: What is a Viable System Model?
Viable System Model (VSM) and Cybernetics. Stafford Beer, etc.
Alternative queries: viable system model, cybernetics, stafford beer, management cybernetics, organizational cybernetics

First explain the query and the desired content, then provide a comma separated list of alternative queries.

# Actual
Query: {query}
"""


  expand_query_prompt = PromptTemplate(input_variables=["query"], template=expand_query_template)
  expand_query_chain = LLMChain(llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.02), prompt=expand_query_prompt, output_key='expand_query_output')
  process_output_chain = TransformChain(input_variables=["expand_query_output"], output_variables=['queries'], transform=process_queries_output)
  sequential_chain = SequentialChain(chains=[expand_query_chain, process_output_chain], input_variables=['query'], output_variables=['queries'])
  return sequential_chain

def search_papers(query):
  print(f"Searching for papers with query: {query}")
  papers = Works().search(query).get(per_page=100)
  return papers

def store_papers(papers):
  papers_abstracts = [f"{p['title']} {p['doi']}\n{p['abstract']}" for p in papers]
  #print(papers_abstracts)
  print(f"Storing {len(papers_abstracts)} abstracts...")
  return vector_store.add_texts(papers_abstracts)

def create_retriever():
  global vector_store

  # Embedding and Context Store
  COLLECTION_NAME = "papers"
  
  CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=os.environ.get("PGVECTOR_HOST"),
    port=int(os.environ.get("PGVECTOR_PORT")),
    database=os.environ.get("PGVECTOR_DATABASE"),
    user=os.environ.get("PGVECTOR_USER"),
    password=os.environ.get("PGVECTOR_PASSWORD"),
  )
  
  print("Getting base embeddings...")
  embeddings = OpenAIEmbeddings()
  print("Embeddings loaded.")
  
  print("Fetching vector store...")
  vector_store = PGVector(
    connection_string=CONNECTION_STRING,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
  )
  
  # Retrieval
  llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

  question_prompt_template = """Use the following scientific paper abstracts to see if any of them is relevant to answer the question. 
Return any relevant text, specially full title and reference url.

Title url
Description.

{context}
Question: {question}
Relevant papers, if any:"""
  QUESTION_PROMPT = PromptTemplate(
      template=question_prompt_template, input_variables=["context", "question"]
  )

  combine_prompt_template = """You are an advanced AI search engine with access to hundreds of scientific papers abstracts. Given the following scientific paper abstracts and a question, create a final answer. 
If you don't know the answer, just say that you don't know. You can use your knowledge, but don't try to make up an answer.
Include the full title and reference url of the papers.

QUESTION: {question}
=========
{summaries}
=========
Answer:"""
  COMBINE_PROMPT = PromptTemplate(
    template=combine_prompt_template, input_variables=["summaries", "question"]
  )
  doc_chain = load_qa_chain(llm, chain_type="map_reduce", question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)

  memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
  question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

  retriever = ConversationalRetrievalChain(
      retriever=vector_store.as_retriever(similarity_top_k=30),
      memory=memory,
      question_generator=question_generator,
      combine_docs_chain=doc_chain
    )

  return retriever    

if __name__ == '__main__':
  
  load_dotenv()
  
  retriever = create_retriever()
  query_expander = create_query_expander()

  while True:

    query = input("# ")
      
    if query == "":
      break
    
    try:
      for sub_query in query_expander.run(query):
        papers = search_papers(sub_query)
        store_papers(papers)

    except Exception as e:
      print(e)
    
    print("Doing a little thinking...")

    print(f"> {retriever({'question': query})['answer']}")
