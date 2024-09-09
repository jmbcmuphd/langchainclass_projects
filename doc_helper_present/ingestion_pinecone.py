import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
    "https://www.saab.com/markets/united-states/skapa-by-saab/",
    "https://www.saab.com/markets/united-states/skapa-by-saab#we-are-skapa---where-dreamers-and-doers-unite",
    "https://www.saab.com/markets/united-states/skapa-by-saab#saab-launches-new-initiative-to-shape-the-future-of-defense-and-security",
    "https://www.saab.com/markets/united-states/skapa-by-saab#our-team",
    "https://www.saab.com/markets/united-states/skapa-by-saab#contact-skapa",
    "https://www.saab.com/markets/united-states/",
    "https://www.saab.com/markets/united-states/about",
    "https://www.saab.com/markets/united-states/about/organization",
    "https://www.saab.com/markets/united-states/about/units",
    "https://www.saab.com/markets/united-states/about/supplier-information",
    "https://www.saab.com/products",
    "https://www.saab.com/products/air",
    "https://www.saab.com/products/land",
    "https://www.saab.com/products/naval",
    "https://www.saab.com/products/security",
    "https://www.saab.com/newsroom",
    "https://www.saab.com/investors",
    "https://www.saab.com/sustainability",
    "https://www.saab.com/markets/united-states/us-newsroom/news-and-press-releases",
    "https://www.saab.com/markets/united-states/us-newsroom/stories/2023/saabs-autonomy-and-ai-team-energizes-smarter-products",
    "https://www.saab.com/markets/united-states/us-newsroom/news-and-press-releases/2024/saab-launches-new-initiative-to-shape-the-future-of-defense-and-security",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=200, chunk_overlap=10
)
doc_splits = text_splitter.split_documents(docs_list)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                              openai_api_key=os.environ.get("OPENAI_API_KEY"))

print("Storing documents in PineCone")
PineconeVectorStore.from_documents(
    doc_splits, embeddings, index_name=os.environ["INDEX_NAME"] )
print("Documents stored in Pinecone")

# retriever = PineconeVectorStore.from_documents(
#     doc_splits, embeddings, index_name=os.environ["INDEX_NAME"]
# ).as_retriever()
