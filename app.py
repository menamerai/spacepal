import streamlit as st
from main import load_pdf_to_text, generate_response, init_chain

"""
# STAR
"""
model_name = "cohere"
key = "6TL2N5ysRGhfd1FiCykEZCXFfyYTZwv9um4LPshX"
chain = init_chain(model_name, key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if 'pdf' not in st.session_state:
    st.session_state['pdf'] = None
if 'submitted' not in st.session_state: 
    st.session_state['submitted'] = False

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state["submitted"]:
    with st.chat_message('ai', avatar='🤖'):
        st.write('Please submit a pdf!')
        uploaded_pdfs = st.file_uploader('Upload a PDF', type=['pdf'])
        submit_button = st.button('Submit', type='primary')
        print(type(uploaded_pdfs))
        if submit_button:
            st.session_state['submitted'] = True
            st.session_state['pdf'] = uploaded_pdfs

if st.session_state['pdf'] is not None:
    text = load_pdf_to_text(st.session_state.pdf.name)

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

