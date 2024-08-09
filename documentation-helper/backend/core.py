import os
from typing import Any

from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Pinecone as PineconeLangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
index_name = os.environ["INDEX_NAME"]
pinecone_api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=pinecone_api_key)


def run_llm2(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeLangChain.from_existing_index(index_name=index_name, embedding=embeddings)
    chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


def run_llm(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    chat = ChatOpenAI(model="gpt-4o-mini", verbose=True, temperature=0)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["answer"])

    result = run_llm2(query="What is a LangChain chain?")
    print(result["result"])
