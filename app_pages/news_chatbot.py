import urllib.parse

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient

load_dotenv()

username = urllib.parse.quote_plus("alvarogutierrezmoraga")
password = urllib.parse.quote_plus("TE(dn93(@7euDAx7b(+A")
MONGODB_ATLAS_CLUSTER_URI = f"mongodb+srv://{username}:{password}@news-scraper.lw0oo.mongodb.net/?retryWrites=true&w=majority&appName=news-scraper"
DB_NAME = "sample_mflix"
COLLECTION_NAME = "movies"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "movies_vector_index"


def get_embedding(text):
    """Generates vector embeddings for the given text."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    return embedding_model.embed_query(text)


if __name__ == "__page__":
    st.write("Hello, world!")

    client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)
    collection = client[DB_NAME][COLLECTION_NAME]

    col_filter = {"fullplot": {"$nin": [None, ""]}}
    documents = collection.find(col_filter).limit(50)
    updated_doc_count = 0
    for doc in documents:
        st.write(doc["fullplot"])
        embedding = get_embedding(doc["fullplot"])
        collection.update_one({"_id": doc["_id"]}, {"$set": {"embedding": embedding}})
        updated_doc_count += 1
    print(f"Updated {updated_doc_count} documents.")
