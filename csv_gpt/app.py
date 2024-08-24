import streamlit as st
from utils import CSVProcessor

st.title("Chat With CSV Files and Text")
st.write("Ask anything about the parts of engine componets")
uploaded_file = st.file_uploader("Upload a csv file", type=["csv"])

if uploaded_file is not None:
    query = st.text_area("Ask any question related to the document")
    button = st.button("Submit")
    if button:
        st.write(CSVProcessor.get_answer_from_llm(uploaded_file, query))
