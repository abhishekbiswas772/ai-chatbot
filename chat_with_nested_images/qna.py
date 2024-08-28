from llm_ops import LLMHandler
from img_loader import ImgHandler
from langchain_core.messages import HumanMessage
import os
import pandas as pd

def remove_asterisks(text):
    if text.startswith('**') and text.endswith('**'):
        return text[2:-2]
    return text

def find_details_from_database(part_name):
    csv_path = "./car_engine_parts.csv"
    df = pd.read_csv(csv_path)
    part_name = part_name.lower()
    
    result = df[df['partname'].str.lower().str.contains(part_name, na=False)]
    
    if not result.empty:
        return result[['id', 'partname' ,'description']]
    else:
        return pd.DataFrame(columns=['id',"partname",'description'], index=None)


class QNAHandler:
    @staticmethod
    def build_prompt(text, main_image, query_image):
        message = HumanMessage(
            content=[
                {"type": "text", "text": "You are an expert algorithm for comparing images. Your task is to identify and find the name of the query image from the text present in the main image."},
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{main_image}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{query_image}"}},
            ]
        )
        return message
       
    
    @staticmethod
    def make_qna(image_file):
        base_path = "./base_images"
        llm = LLMHandler.load_llm_model()
        original_image = ImgHandler.load_query_image_and_embedding(file_path=image_file)
        print("******** Result from milvus ************")
        print(original_image)
        print("******************************************")
        text = "Give me the name of the component in the query image by seeing the main image. and only give Name eg: **<NAME>**"
        full_base_img_path = os.path.join(base_path, original_image)
        full_query_path = image_file
        main_image = ImgHandler.load_image(full_base_img_path)["image"]
        query_image = ImgHandler.load_image(full_query_path)["image"]
        prompt = QNAHandler.build_prompt(text, main_image, query_image)
        response = llm.invoke([prompt])
        result = remove_asterisks(response.content)
        df = find_details_from_database(result.lower())
        return df
            
