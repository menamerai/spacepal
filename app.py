import streamlit as st
from main import load_pdf_to_text, generate_response, init_chain

"""
# STAR
"""
model_name = "teknium/Phi-Hermes-1.3B"
key = "6TL2N5ysRGhfd1FiCykEZCXFfyYTZwv9um4LPshX"
chain = init_chain(model_name, key)

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
    # if True: # do flag here
    if prompt := st.chat_input("Say Something"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        # st.session_state['submitted'] = st.button('Submit', type='primary')
        response = f"{generate_response(prompt, chain)}"
        # Display assistant response in chat message container
        with st.chat_message("ai", avatar='🤖'):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})            

