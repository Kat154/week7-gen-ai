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
from flask import Flask
from flask import Flask, request,redirect, url_for, send_from_directory, render_template
# defining our project specific variables:

PINECONE_VECTOR_DB_NAME = 'final-vector-db'
# ---------------------------------
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index(PINECONE_VECTOR_DB_NAME)

client = OpenAI()
# useful functions:----------------------------------------------------
''' to know more about our vector database use the following methods:
index.describe_index_stats()


'''
def get_embeddings(text, model = "text-embedding-ada-002"):
    text = text.replace("\n"," ")
    return client.embeddings.create(input = text, model = model).data[0].embedding

def get_context(query, embed_model='text-embedding-ada-002', k=10):
    query_embeddings = get_embeddings(query,model=embed_model)
    pinecone_res = index.query(vector = query_embeddings, top_k = k, include_metadata=True)
    contexts = [item['metadata']['text'] for item in pinecone_res['matches']]
    # print(f'context:{contexts}')
    document_names = [item['metadata']['document_name'] for item in pinecone_res['matches']]
    return contexts,document_names, query

def augmented_query(user_query,embed_model = 'text-embedding-ada-002',k=5):
    contexts,document_names, query = get_context(user_query, embed_model = embed_model, k=k)
    # print("\n\n--\n\n".join(contexts)+"\n\n--\n\n"+query)
    return "\n\n--\n\n".join(contexts)+"\n\n--\n\n"+query

def ask_gpt(system_prompt,user_prompt, model="gpt-3.5-turbo"):
    temperature_=0.7
    completion = client.chat.completions.create(
    model=model,
    temperature=temperature_,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    )
    lines = (completion.choices[0].message.content).split("\n")
    lists = (textwrap.TextWrapper(width=90, break_long_words=False).wrap(line) for line in lines)
    return "\n".join("\n".join(list) for list in lists)
# primer = "if you do not know the answer say i dont know and dont make up stuff"
# print(ask_gpt(system_prompt=primer, user_prompt=augmented_query('what is AWS service')))

def normal_AI(query):
    embed_model = 'text-embedding-ada-002'
    primer = f"""
    You are a Question and Answer bot. A highly intelligent system that answers
    user questions based on information provided by the user above each question.
    If the answer cannot be found in the information provided by the user, you truthfully
    answer "I don't know"
    """
    llm_model = 'gpt-3.5-turbo'
    user_prompt = augmented_query(query,embed_model)
    return ask_gpt(primer, user_prompt,model=llm_model)

def advanced_AI(query):
    embed_model = 'text-embedding-ada-002'
    primer = f"""
    You are a Question and Answer bot. A highly intelligent system that answers
    user questions based on information provided by the user above each question.
    If the answer cannot be found in the information provided by the user, you truthfully
    answer "I don't know"
    """
    # primer = f"""
    # you are learned individual who will answer  question in elaborate manner  if you do not know the answer you will say you dont know. . you will not make up stuff and will only answer if you know """
    llm_model = 'gpt-4-turbo-2024-04-09'
    user_prompt = augmented_query(query,embed_model)
    return ask_gpt(primer, user_prompt,model=llm_model)




# ---------------------Flask logic below------------------------

app = Flask(__name__)
@app.route("/")
def template_test():
    return render_template('index.html', label='', imagesource='file://null')


@app.route('/analyze', methods=['POST'])
def take_text_input_from_user():

  our_text = request.form['userInput']
  predicted_value = normal_AI(our_text)
  all_sources=list(set(get_context(our_text)[1]))
  final_str=''
  for current_value in all_sources:
    final_str+=' '+current_value
    final_str+= ', '
  final_str=final_str[:-2]
  return render_template("index.html", result=predicted_value,sources=final_str)

if __name__=="__main__":
    app.run()
