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
import textwrap 
# defining our project specific variables:

PINECONE_VECTOR_DB_NAME = 'viral-news'
# ---------------------------------
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(PINECONE_VECTOR_DB_NAME)

client = OpenAI()
# print('here')
# useful functions:----------------------------------------------------
''' to know more about our vector database use the following methods:
index.describe_index_stats()


'''
def get_embeddings(text, model = "text-embedding-ada-002"):
  text = text.replace("\n"," ")
  return client.embeddings.create(input = text, model = model).data[0].embedding

def get_context(query, embed_model='text-embedding-ada-002', k=5):
  query_embeddings = get_embeddings(query,model=embed_model)
  pinecone_res = index.query(vector = query_embeddings, top_k = k, include_metadata=True)
  contexts = [item['metadata']['text'] for item in pinecone_res['matches']]
  page_numbers = [item['metadata']['page_numbers'] for item in pinecone_res['matches']]
  document_name = [item['metadata']['document_name'] for item in pinecone_res['matches']]
  return [contexts,page_numbers,document_name], query

