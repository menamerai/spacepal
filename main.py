import torch
import argparse

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from pypdf import PdfReader

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.llms.cohere import Cohere
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback

from typing import Optional

import pickle
import os
import base64

model_name = "teknium/Phi-Hermes-1.3B"
# model_name = "teknium/Puffin-Phi-v2"
device = 0 if torch.cuda.is_available() else -1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="teknium/Phi-Hermes-1.3B")
    parser.add_argument("-k", "--key", type=str)
    args = parser.parse_args()
    if args.model_name == "cohere" and args.key is None:
        raise ValueError("must provide API key if using cohere backend")
    return args

def load_pdf_to_text(pdf_path: str): 
    pdf_reader = PdfReader(pdf_path)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def rag_process(pdf_path: str, text: str, model_name: str, question: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 40,
        length_function = len
    )
    chunks = text_splitter.split_text(text=text)

    store_name = pdf_path[:-4]
        
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            vectorstore = pickle.load(f)
        #st.write("Already, Embeddings loaded from the your folder (disks)")
    else:
        #embedding (Openai methods) 
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

        #Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(vectorstore,f)

    retriever = vectorstore.as_retriever()
    prompt_template = "### Instruction: Use the following context to answer the question. Base your answers on the context given, do not make up information.\nContext:\n{context}\nQuestion: {question}\n### Response: \n"
    template = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        device=device,  # -1 for CPU
        batch_size=1,  # adjust as needed based on GPU map and model size.
        model_kwargs={"do_sample": True, "temperature": 0.8, "max_length": 2048, "torch_dtype": torch.bfloat16, "trust_remote_code": True},
    )
    chain_type_kwargs = {"prompt": template}
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    
    # question = input("Input Question: ")
    response = chain.run(question)

    return response

def init_chain(model_name: str, key: Optional[str] = None) -> RetrievalQA:
    pdf_path = "test.pdf"
    text = load_pdf_to_text(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 200,
        chunk_overlap = 40,
        length_function = len
    )
    chunks = text_splitter.split_text(text=text)

    store_name = pdf_path[:-4]
        
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl","rb") as f:
            vectorstore = pickle.load(f)
        #st.write("Already, Embeddings loaded from the your folder (disks)")
    else:
        #embedding (Openai methods) 
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

        #Store the chunks part in db (vector)
        vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

        with open(f"{store_name}.pkl","wb") as f:
            pickle.dump(vectorstore,f)

    retriever = vectorstore.as_retriever()
    
    if model_name == "cohere":
        llm = Cohere(cohere_api_key=key)
        prompt_template = "You are a superintelligent AI assistant that excels in handling technical documents. Use the following context to answer the question. Base your answers on the context given, do not make up information. If you don't know something, just say it.\nContext:\n{context}\nQuestion: {question}\nAnswer: "
    else:
        prompt_template = "### Instruction: Use the following context to answer the question. Base your answers on the context given, do not make up information.\nContext:\n{context}\nQuestion: {question}\n### Response: \n"
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device=device,  # -1 for CPU
            batch_size=1,  # adjust as needed based on GPU map and model size.
            model_kwargs={"do_sample": True, "temperature": 0.8, "max_length": 2048, "torch_dtype": torch.bfloat16, "trust_remote_code": True},
        )

    template = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    chain_type_kwargs = {"prompt": template}
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    return chain

def main(chain: RetrievalQA): 
    question = input("Input Question: ")
    response = chain.run(question)

    print(response)

if __name__ == "__main__":
    args = parse_args()
    chain = init_chain(args.model_name, args.key)
    main(chain) 


    