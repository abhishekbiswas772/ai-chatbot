import streamlit as st
import os
from tempfile import NamedTemporaryFile
from processor import image_rag

st.title("Chat With Image")
option = st.selectbox("Choose the chat mode:", ("Text", "Image"))

if option == "Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    text_input = st.text_input("Enter your message:")
    if st.button("Send Image"):
        if uploaded_file is not None:
            with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.read())
                st.image(temp_file.name, caption="Uploaded Image", use_column_width=True)
                result = image_rag(text_input, temp_file.name)
                st.subheader("Result")
                st.write(result)
        else:
            st.error("Please upload an image first.")
elif option == "Text":
    text_input = st.text_input("Enter your message:")
    if st.button("Send Text"):
        result = image_rag(text_input, None)
        st.subheader("Result")
        st.write(result)
    

