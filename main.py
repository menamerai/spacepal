import re
import torch
import argparse

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from pypdf import PdfReader
from pathlib import Path
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.llms.cohere import Cohere
from langchain.chains import RetrievalQA

from typing import Optional, List, Dict, Tuple, Any

import pickle
import os
import base64

model_name = "teknium/Phi-Hermes-1.3B"
# model_name = "teknium/Puffin-Phi-v2"
device = 0 if torch.cuda.is_available() else -1

spec_toc_pattern = "[0-9]+\.[0-9\.]*\s?[a-z A-Z0-9\-\,\(\)\n\?]+\s?[\.\s][\.\s]+\s?[0-9]+"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-name", type=str, default="teknium/Phi-Hermes-1.3B")
    parser.add_argument("-k", "--key", type=str)
    parser.add_argument("-p", "--path", type=str, required=True)
    args = parser.parse_args()
    if args.model_name == "cohere" and args.key is None:
        raise ValueError("must provide API key if using cohere backend")
    return args

def get_toc_pages(pages: List) -> List[Any]:
    state = 0 # 0 means we are looking for TOC start, 1 means we're parsing regex
    i = 0
    toc_pages = []
    while True:
        page = pages[i]
        text = page.extract_text()
        match state:
            case 0:
                if "table of contents" in text.lower():
                    state = 1
                else:
                    i += 1
            case 1:
                matches = re.findall(spec_toc_pattern, text)
                if len(matches) > 0:
                    toc_pages += [page]
                else:
                    break
                i += 1

    max_pg_number = 0
    for i, page in enumerate(toc_pages):
        text = page.extract_text()
        matches = [x for x in re.findall(spec_toc_pattern, text)]
        pg_number_pattern = "[0-9]+$"
        pg_number_strs = [re.findall(pg_number_pattern, match.strip()) for match in matches]
        pg_numbers = [int(num[0]) for num in pg_number_strs if len(num) > 0] + [0]
        if max(pg_numbers) < max_pg_number:
            return toc_pages[:i] 
        else:
            max_pg_number = max(pg_numbers)

    return toc_pages

def get_spec_entry(pages: List, title: str, entries: Dict[str, Dict]) -> Tuple[int]:
    entry = entries[title]
    pg_num = entry['page_number'] - 1
    spec_num, spec_title = entry['spec_number'], entry['spec_title']
    for page in pages[pg_num:]:
        content = page.extract_text()
        content_lines = [line.strip() for line in content.split("\n") if line.strip() != ""]

        state = 0
        spec_content = ""
        for line in content_lines:
            match state:
                case 0:
                    if spec_num in line and spec_title in line:
                        spec_content += (line + "\n")
                        state = 1
                case 1:
                    found_titles = [t for t, v in entries.items() if v['spec_number'] in line and v['spec_title'] in line and t != title]
                    if len(found_titles) == 0:
                        spec_content += (line + "\n")                    
                    else:
                        break
        if spec_content != "":
            return spec_content
    return None

def get_spec_list(pdf_path: str, verbose: bool = False) -> List[str]:
    pdf_reader = PdfReader(pdf_path)
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def binary_to_pdf(bin, dir: str, name: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(Path(dir) / name, "wb") as pdf_out:
        pdf_out.write(bin)

@st.cache_resource
def init_chain(model_name: str, key: Optional[str] = None) -> RetrievalQA:
    pdf_path = "test1.pdf"
    text = load_pdf_to_text(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len
    )
    for i in idx:
        specs += text_splitter.split_text(specs[i])
    return specs

def binary_to_pdf(bin, dir: str, name: str):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(Path(dir) / name, "wb") as pdf_out:
        pdf_out.write(bin)

@st.cache_resource
def init_chain(model_name: str, pdf_path: str, chunk_size: int = 1000, key: Optional[str] = None) -> RetrievalQA:
    # pdf_path = "test1.pdf"
    # text = load_pdf_to_text(pdf_path)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 200,
    #     chunk_overlap = 40,
    #     length_function = len
    # )
    # chunks = text_splitter.split_text(text=text)

    chunks = load_pdf_to_chunks(pdf_path, chunk_size=chunk_size, verbose=True)
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
        prompt_template = "### Instruction: You are an AI assistant NASA missions used specifically to proof-read their documentations. Use the following context to answer the question. Base your answers on the context given, do not make up information.\nContext:\n{context}\nQuestion: {question}\n### Response: \n"
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

def generate_response(question: str, chain: RetrievalQA):
    response = chain.run(question)
    return response

def main(chain: RetrievalQA): 
    question = input("Input Question: ")
    response = chain.run(question)

    print(response)

if __name__ == "__main__":
    args = parse_args()
    chain = init_chain(args.model_name, args.path, key=args.key)
    main(chain) 
    