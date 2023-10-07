import streamlit as st


"""
# STAR
"""

with st.chat_message('ai', avatar='🤖'):
    st.write('Please submit a pdf!')
    st.session_state['pdf'] = st.file_uploader('Upload a PDF', type=['pdf'])

    if st.session_state['pdf'] is not None:
        # load to model
        
        # set some analyzed flag to true

        if True: # do flag here
            st.text_area('Prompt')
            st.session_state['submitted'] = st.button('Submit', type='primary')
