import os, re, json
import numpy as np
from datetime import datetime
# from sentence_transformerclss import SentenceTransformer
from pymilvus import connections, utility, CollectionSchema, DataType, FieldSchema, Collection
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
# from app.settings import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_COLLECTION_NAME

DB_USER = "minioadmin"
DB_PASSWORD = "minioadmin"
DB_HOST = "localhost"
DB_PORT = "19530"
DB_COLLECTION_NAME = "partEmbeddingEngine"
model = resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    image = Image.open(image_path).convert('RGB') 
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    # return features.squeeze().numpy()
    return features.squeeze().numpy().astype(np.float32)
 


def connect_to_milvus():
    try:
        connections.connect("default",user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        print("Connected to Milvus.")
        return 1
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_collection(name, fields, description, consistency_level="Strong"):
    try:
        schema = CollectionSchema(fields, description)
        collection = Collection(name, schema, consistency_level=consistency_level)
        print(f"Collection '{name}' created.")
        return collection
    except Exception as e:
        print(f"Failed to create collection: {e}")
        return None

def insert_data(collection, entity):
    try:
        data = [
            [entity['pid']],  # Wrap pid in a list
            [entity['vector']] # Convert numpy array to list
        ]
        collection.insert(data)
        collection.flush()
        print(f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")

        return {"collection_name" : collection.name, "collection_num_entities" : collection.num_entities}

    except Exception as e:
        print(f"Failed to insert data: {e}")
        return None


def create_index(collection, field_name, index_type, metric_type, params):
    try:
        index = {"index_type": index_type, "metric_type": metric_type, "params": params}
        collection.create_index(field_name, index)
        print(f"Index '{index_type}' created for field '{field_name}'.")
    except Exception as e:
        print(f"Failed to create index: {e}")

def insert_into_db(entity, dims):
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=90, is_primary=True, auto_id=True),
        FieldSchema(name="pid", dtype=DataType.VARCHAR, max_length=90),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim= dims)
    ]
    collection = create_collection(DB_COLLECTION_NAME, fields, "Collection for Dev Milvus")

    if collection is not None:
        result = insert_data(collection, entity)
        print(">>> result == ",result)
        return [1, result]
    else:
        print("Collection creation failed. Aborting further operations.")
        return [0,"Collection creation failed"]

def save_results_to_file(results, val="result"):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    current_date = datetime.now().strftime("%d%m%Y")
    timestamp = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    filename = f"logs/{current_date}.log"
    if (val=="query") : 
        with open(filename, "a") as file:
            file.write(f"{timestamp} - Query :\n")
            file.write(json.dumps(results, indent=4))
            file.write("\n")
    else :
        print(f"Storing results to {filename}")
        with open(filename, "a") as file:
            file.write(f"{timestamp} - Searched results are:\n")
            file.write(json.dumps(results, indent=4))
            file.write("\n\n")

# def search_and_query(collection, query_embeddings):
#     try:
#         res = collection.search(
#             data=query_embeddings,
#             anns_field="vector",
#             param={"metric_type": "L2", "params": {}},
#             limit=10,
#             expr=None
#         )
#         filtered_results = []
#         # print(">>> Searched response : ", res)
#         # return [1, res]
#         for hits in res:
#             for hit in hits:
#                 if hit.distance > 0.0 and hit.distance <= 1.0 :
#                     filtered_results.append({
#                         "id": hit.id,
#                         "matching": str((1 - round(hit.distance, 2))*100) + "%"
#                     })
        
#         if filtered_results is not None:
#             save_results_to_file(filtered_results)
#             return filtered_results
#         else:
#             print("No results matching beyond 80%")
#             return ["No results matching beyond 80%"]
#     except Exception as e:
#         print(f"Search failed: {e}")
#         return ["No results matching beyond 80%"]

# def semantic_search(entity):
#     try:
#         collection = Collection(name=DB_COLLECTION_NAME)
#         create_index(collection, "vector", "IVF_FLAT", "L2", {"nlist": 128})
#         collection.load()
#         print(f"Collection '{collection.name}' loaded successfully.")
#         # print(">>>> entity ",entity["vector"])
#         result = search_and_query(collection, entity["vector"])
#         return [1, result]
#     except Exception as e:
#         print(f"Failed to load collection: {e}")
#         return [0, e]


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

        if filtered_results:
            save_results_to_file(filtered_results)
            return filtered_results
        else:
            print("No results matching beyond 80%")
            return ["No results matching beyond 80%"]
    except Exception as e:
        print(f"Search failed: {e}")
        return ["Search failed due to an error"]


def semantic_search(entity):
    try:
        collection = Collection(name=DB_COLLECTION_NAME)
        create_index(collection, "vector", "IVF_FLAT", "L2", {"nlist": 128})
        collection.load()
        print(f"Collection '{collection.name}' loaded successfully.")

        # Ensure the vector is a 2D array for Milvus search
        query_embeddings = np.array(entity["vector"], dtype=np.float32).reshape(1, -1)
        result = search_and_query(collection, query_embeddings)
        return [1, result]
    except Exception as e:
        print(f"Failed to load collection or perform search: {e}")
        return [0, str(e)]


def drop_collection(collection_name):
    try:
        connections.connect("default",user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        print("Connected to Milvus.")
        utility.drop_collection(collection_name)
        print(f"Dropped collection '{collection_name}'.")
        return 1
    except Exception as e:
        print(f"Failed to drop collection: {e}")


def main1():
    connect_to_milvus()
    drop_collection(DB_COLLECTION_NAME)
    image_path = "./images"
    xls_path = "./description.xlsx"
    data_frame = pd.read_excel(xls_path)
    if not os.path.exists(image_path):
        print(f"The directory {image_path} does not exist.")
        return
    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            try:
                features = extract_features(file_path)
                entity = {"pid": filename, "vector": features}
                shape_len = entity['vector'].shape
                status = insert_into_db(entity=entity, dims=shape_len[0])
                print(status)
            except Exception as e:
                print(e)


def main2():
    connect_to_milvus()
    query_path = "./query"
    xls_path = "./description.xlsx"
    data_frame = pd.read_excel(xls_path)
    if not os.path.exists(query_path):
        print(f"The directory {query_path} does not exist.")
        return 
    for filename in os.listdir(query_path):
        file_path = os.path.join(query_path, filename)
        features = extract_features(file_path)
        entity = {"vector": features}
        search_status = semantic_search(entity)
        print(search_status)
    


if __name__ == "__main__":
    main2()