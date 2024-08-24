import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from pymilvus import connections, utility, Collection
import pandas as pd
import os

# Define constants for Milvus connection
DB_USER = "minioadmin"
DB_PASSWORD = "minioadmin"
DB_HOST = "localhost"
DB_PORT = "19530"
DB_COLLECTION_NAME = "partEmbeddingEngine"

# Load the pre-trained model
model = resnet18(pretrained=True)
model.eval()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image):
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy().astype(np.float32)

# Function to connect to Milvus
def connect_to_milvus():
    connections.connect("default", user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)

# Function to search and query in Milvus
def search_and_query(collection, query_embeddings):
    try:
        # Perform the search
        res = collection.search(
            data=query_embeddings,
            anns_field="vector",
            param={"metric_type": "L2", "params": {}},
            limit=10,
            expr=None,
            output_fields=["pid"]  # Ensure 'pid' is retrieved in search results
        )

        filtered_results = []
        for hits in res:
            for hit in hits:
                if 0.0 <= hit.distance < 1.0:
                    entity_pid = hit.entity.get('pid')
                    filtered_results.append({
                        "id": hit.id,
                        "matching": f"{(1 - hit.distance) * 100:.0f}%",  # Correct calculation for matching percentage
                        "pid": entity_pid
                    })

        return filtered_results
    except Exception as e:
        print(f"Search failed: {e}")
        return None

# Function to perform semantic search
def semantic_search(entity):
    try:
        collection = Collection(name=DB_COLLECTION_NAME)
        collection.load()
        print(f"Collection '{collection.name}' loaded successfully.")

        # Ensure the vector is a 2D array for Milvus search
        query_embeddings = np.array(entity["vector"], dtype=np.float32).reshape(1, -1)
        result = search_and_query(collection, query_embeddings)
        return result
    except Exception as e:
        print(f"Failed to load collection or perform search: {e}")
        return None

def main():
    st.title("Image Search")

    # Load the Excel file with descriptions
    xls_path = "./description.xlsx"
    if not os.path.exists(xls_path):
        st.error("The description file does not exist.")
        return

    data_frame = pd.read_excel(xls_path)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Extracting features and getting information...")
        features = extract_features(image)
        entity = {"vector": features}
        connect_to_milvus()
        search_results = semantic_search(entity)
        if search_results:
            st.subheader("Search Results")
            for result in search_results:
                # search_key = result['pid'].replace('.jpg', '')
                search_key = os.path.splitext(result['pid'])[0]
                print(search_key)
                matching_row = data_frame[data_frame['id'] == search_key]
                if not matching_row.empty:
                    res = matching_row['id'].values[0], matching_row['description'].values[0]
                    # st.write(f"ID: {res[0]}")
                    st.write(f"Description: {res[1]}")
                else:
                    st.write(f"No description found for ID: {search_key}")
        else:
            st.write("No results matching beyond 80%")

if __name__ == "__main__":
    main()