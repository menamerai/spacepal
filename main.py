import re
import torch
import argparse

import numpy as np
import streamlit as st

from streamlit_extras.add_vertical_space import add_vertical_space
from pypdf import PdfReader

from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.llms.cohere import Cohere
from langchain.chains import RetrievalQA

from transformers import pipeline

from typing import Optional, List, Dict, Tuple, Any, Union

import pickle
import os
import base64

Model = Dict[str, Union[Any,  RetrievalQA]]

model_name = "teknium/Phi-Hermes-1.3B"
nli_model = "sileod/deberta-v3-base-tasksource-nli"
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

def get_spec_list(pdf_path: str, verbose: bool = False) -> Tuple[List[str], Dict[str, Dict]]:
    pdf_reader = PdfReader(pdf_path)
    entries = get_toc_entries(pdf_reader.pages)

    specs = []
    entry_iter = entries.keys()
    if verbose:
        entry_iter = tqdm(entry_iter, total=len(entries))
    for title in entry_iter:
        specs += [get_spec_entry(pdf_reader.pages, title, entries)]
    return [s for s in specs if s is not None], entries

def get_toc_entries(pages: List) -> Dict[str, Dict]:
    pages = get_toc_pages(pages)
    split_pattern = "\.\.+" # match 2 or more dots
    entries = {}
    for page in pages:
        text = page.extract_text()
        matches = [x for x in re.findall(spec_toc_pattern, text)]
        for match in matches:
            match_components = re.split(split_pattern, match)
            match_components[0] = match_components[0].replace("\n", "").strip()
            match_components[-1] = match_components[-1].split()[0].replace(".", "").strip()
            tokens = match_components[0].split() 
            spec_num, spec_title = tokens[0], ' '.join(tokens[1:])

            pg_num = int(match_components[-1])
            entries[match_components[0]] = {
                "page_number": pg_num,
                "spec_number": spec_num,
                "spec_title": spec_title
            }
    return entries

def load_pdf_to_chunks(pdf_path: str, chunk_size: Optional[int] = None, verbose: bool = False) -> List[str]:
    specs, toc_entries = get_spec_list(pdf_path, verbose=verbose) 
    if chunk_size is not None:
        lens = [len(s) for s in specs]
        idx = [i for i, l in enumerate(lens) if l > chunk_size]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=40,
            length_function=len
        )
        for i in idx:
            specs += text_splitter.split_text(specs[i])
    return specs, toc_entries

def binary_to_pdf(bin, dir: str):
    with open(os.path.expanduser(dir), "wb") as pdf_out:
        pdf_out.write(base64.b64decode(bin))

@st.cache_resource
def init_chain(model_name: str, pdf_path: str, chunk_size: int = 1000, key: Optional[str] = None) -> Tuple[Model, Dict[str, Dict]]:
    # pdf_path = "test1.pdf"
    # text = load_pdf_to_text(pdf_path)
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 200,
    #     chunk_overlap = 40,
    #     length_function = len
    # )
    # chunks = text_splitter.split_text(text=text)

    chunks, toc_entries = load_pdf_to_chunks(pdf_path, verbose=True)
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

    classifier = pipeline("zero-shot-classification", model=nli_model, device=device)
    return {"chain": chain, "classifier": classifier}, toc_entries

def generate_response(question: str, model: Model, toc_entries: Dict[str, Dict] = {}):
    if len(toc_entries) > 0:
        toc_labels = list(toc_entries.keys())
        results = model['classifier'](question, toc_labels)
        idxs = np.argsort(results['scores'])[::-1][:10]
        for i in idxs:
            print(results['labels'][i] + " -- " + str(results['scores'][i]))

    response = model['chain'].run(question)
    return response

def main(chain: RetrievalQA): 
    question = input("Input Question: ")
    response = chain.run(question)

    print(response)

if __name__ == "__main__":
    args = parse_args()
    model, toc_entries = init_chain(args.model_name, args.path, key=args.key, chunk_size=2000)
    question = input("Input Question: ")
    response = generate_response(question, model, toc_entries)
    print(response)
    # main(chain) 
    