import os
import streamlit as st
import tempfile
import shutil
import os
from pathlib import Path
from main import load_pdf_to_text, generate_response, init_chain, binary_to_pdf
from dotenv import load_dotenv

"""
# Space PAL
"""
load_dotenv()
model_name = "cohere"
key = os.environ.get("COHERE_API_KEY")

if 'temp_dir' not in st.session_state:
    st.session_state["temp_dir"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if 'pdf' not in st.session_state:
    st.session_state['pdf'] = None
if 'submitted' not in st.session_state: 
    st.session_state['submitted'] = False



if not st.session_state["temp_dir"]:
    temp_dir = tempfile.TemporaryDirectory(prefix="temp")
    st.session_state["temp_dir"] = temp_dir

try:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state["submitted"]:
        with st.chat_message('ai', avatar='🤖'):
            st.write('Please submit a pdf!')
            uploaded_pdfs = st.file_uploader('Upload a PDF', type=['pdf'])
            submit_button = st.button('Submit', type='primary')
            if submit_button:
                st.session_state['submitted'] = True
                st.session_state['pdf'] = uploaded_pdfs
                temp_path = "." + str(Path(st.session_state["temp_dir"].name))
                pdf_name = st.session_state['pdf'].name
                pdf_path = Path(temp_path) / pdf_name
                binary_to_pdf(st.session_state['pdf'].read(), str(temp_path), pdf_name) 
                text = load_pdf_to_text(pdf_path) # CHAIN HERE
                chain = init_chain(model_name, key)


    if st.session_state['pdf'] is not None:
        if prompt := st.chat_input("Say Something"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            response = f"{generate_response(prompt, chain)}"

            # Display assistant response in chat message container
            with st.chat_message("ai", avatar='🤖'):
                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})          
finally: 
    st.stop()
    if st.session_state['temp_dir']:
        if os.path.exists("tmp"):
            shutil.rmtree("tmp")

