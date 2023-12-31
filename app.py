import os
import streamlit as st
import tempfile
import shutil
import os
import time
from pathlib import Path
from main import generate_response, init_chain, binary_to_pdf
from dotenv import load_dotenv

st.header('SPACE PAL :rocket:', divider='rainbow')

# Load the API Key
load_dotenv()
model_name = "cohere"
key = os.environ.get("COHERE_API_KEY")

# Initialize session state
if 'temp_dir' not in st.session_state:
    st.session_state["temp_dir"] = False
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if 'pdf' not in st.session_state:
    st.session_state['pdf'] = None
if 'submitted' not in st.session_state: 
    st.session_state['submitted'] = False
if "model" not in st.session_state: 
    st.session_state['model'] = None
if "toc_entries" not in st.session_state: 
    st.session_state['toc_entries'] = None

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
                # Write PDF to temporary file
                time.sleep(1)
                st.session_state['submitted'] = True
                st.session_state['pdf'] = uploaded_pdfs
                temp_path = "." + str(Path(st.session_state["temp_dir"].name))
                pdf_name = st.session_state['pdf'].name
                pdf_path = Path(temp_path) / pdf_name
                binary_to_pdf(st.session_state['pdf'].read(), str(temp_path), pdf_name)
                # Initialize chain for prompt processing 
                st.session_state.model, st.session_state.toc_entries = init_chain(model_name, str(pdf_path), key=key)


    if st.session_state['pdf'] is not None:
        if prompt := st.chat_input("Say Something"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("Generating Response..."):
                response = f"{generate_response(prompt, st.session_state.model, st.session_state.toc_entries)}"

            # Display assistant response in chat message container
            with st.chat_message("ai", avatar='🤖'):
                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})          

# Delete all the temporary files after usage
finally: 
    if st.session_state['pdf']:
        if os.path.exists("./tmp"):
            shutil.rmtree("./tmp")
