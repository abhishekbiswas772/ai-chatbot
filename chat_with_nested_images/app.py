import streamlit as st
from PIL import Image
from main import main2, main1
import tempfile


st.title("Image Upload and Display with Save Path")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        saved_image_path = temp_file.name
        df = main2(file_path=saved_image_path)
        st.write("CSV Part Name Information:")
        st.dataframe(df)

st.sidebar.header("Insert Images to Milvus")
if st.sidebar.button("Start Insertion Data"):
    main1()
