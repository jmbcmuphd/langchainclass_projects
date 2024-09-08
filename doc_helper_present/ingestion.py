import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader(
        "langchain-docs/langchain.readthedocs.io/en/latest", encoding="utf8"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs/", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    index_name = os.environ["INDEX_NAME"]
    PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
    print("**** Loading to vectorstore done *****")


def ingest_docs2() -> None:
    from langchain_community.document_loaders import FireCrawlLoader

    langchain_documents_base_urls = [
        "https://www.saab.com/markets/united-states/",
        "https://www.saab.com/markets/united-states/skapa-by-saab/",
    ]

    for url in langchain_documents_base_urls:
        print(f"FireCrawling {url=}")
        loader = FireCrawlLoader(
            url=url,
            mode="crawl",
            params={
                "crawlerOptions": {"limit": 10},
                "pageOptions": {"onlyMainContent": True},
                "wait_until_done": True,
            },
        )
        docs = loader.load()

        print(f"Going to add {len(docs) }documents to Pinecone")
        PineconeVectorStore.from_documents(docs, embeddings, index_name="saab-usa-skapa-index")
        print(f"****Loading {url}* to vectorstore done ***")



if __name__ == "__main__":
    ingest_docs2()
