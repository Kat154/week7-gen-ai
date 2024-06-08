import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pymupdf4llm
# import fitz
import pymupdf
import pymupdf4llm
import pathlib
from langchain.text_splitter import MarkdownTextSplitter
import os
from openai import OpenAI
from pinecone import Pinecone
# defining our project specific variables:

PINECONE_VECTOR_DB_NAME = 'viral_new'
# ---------------------------------
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(PINECONE_VECTOR_DB_NAME)
client = OpenAI()

def get_embeddings(text, model = "text-embedding-ada-002"):
  text = text.replace("\n"," ")
  return client.embeddings.create(input = text, model = model).data[0].embedding

def get_context(query, embed_model='text-embedding-ada-002', k=5):
  query_embeddings = get_embeddings(query,model=embed_model)
  pinecone_res = index.query(vector = query_embeddings, top_k = k, include_metadata=True)
  contexts = [item['metadata']['text'] for item in pinecone_res['matches']]
  return contexts, query

