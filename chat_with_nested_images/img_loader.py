import os
from embedding_image import EmbeddingHandler
import base64
from milvus_ops import MilvisHandler, DB_COLLECTION_NAME

emb = EmbeddingHandler()


class ImgHandler:
    @staticmethod
    def load_base_image_and_embedding():
        MilvisHandler.connect_to_milvus()
        MilvisHandler.drop_collection(DB_COLLECTION_NAME)
        base_img_path = "./base_images"
        for filename in os.listdir(base_img_path):
            file_path = os.path.join(base_img_path, filename)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                try:
                    features = emb.create_embedding(file_path)
                    entity = {"pid": filename, "vector": features}
                    shape_len = entity['vector'].shape
                    status = MilvisHandler.insert_into_db(entity=entity, dims=shape_len[0])
                    print(status)
                except Exception as e:
                    print(e)


    @staticmethod
    def load_query_image_and_embedding(file_path):
        MilvisHandler.connect_to_milvus()
        features = emb.create_embedding(file_path)
        entity = {"vector": features}
        search_status = MilvisHandler.semantic_search(entity)
        search_status = search_status[-1][0]['pid']
        return search_status
    

    @staticmethod
    def load_image(image_path) -> dict:
        """Load image from file and encode it as base64."""
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        image_base64 = encode_image(image_path)
        return {"image": image_base64}


        