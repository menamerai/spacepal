import streamlit as st
from main import load_pdf_to_text, rag_process

"""
# STAR
"""
model_name = "teknium/Phi-Hermes-1.3B"

with st.chat_message('ai', avatar='🤖'):
    st.write('Please submit a pdf!')
    if 'pdf' not in st.session_state:
        st.session_state['pdf'] = None
    if 'submitted' not in st.session_state: 
        st.session_state['submitted'] = False
        
    uploaded_vids = st.file_uploader('Upload a PDF', type=['pdf'])
    st.session_state['pdf'] = uploaded_vids
    if st.session_state['pdf'] is not None:
        # load to model
        # set some analyzed flag to true
        text = load_pdf_to_text(uploaded_vids.name)
        if True: # do flag here
            prompt = st.text_area('Prompt')
            st.session_state['submitted'] = st.button('Submit', type='primary')
            print(st.session_state)
            if st.session_state['submitted']:
                st.text(rag_process(uploaded_vids.name, text, model_name, prompt))

