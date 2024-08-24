from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from llm_build import LLmBuilder
from custom_agents import get_part_details_from_db
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain_core.prompts.image import ImagePromptTemplate
import base64


def build_prompt(img_path):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an advanced algorithm specialized in image analysis and data retrieval. Your task is to analyze a given image, extract key features and metadata, and then query a database to find relevant information related to the image content. Your analysis should consider various elements such as objects, colors, textures, patterns, and context within the image. After processing the image, formulate precise queries to retrieve data that matches or provides additional insights about the detected features. Focus on accuracy, relevance, and speed in both image analysis and database querying."
            ),
            ("user", "{input}"),
            ("user", f"This is the main image: {img_path}"),
            MessagesPlaceholder(variable_name="agent_db_retriever"),
        ]
    )
    return prompt


def encode_image(image_path):
    """Load image from file and encode it as base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_rag(user_question, image_path):
    llm = LLmBuilder.build_llm()
    result = None
    # image_path = "./image.png"
    prompt = build_prompt(image_path)
    if image_path is not None:
        image_base64 = encode_image(image_path)
    tools = [get_part_details_from_db]
    llm_with_tools = llm.bind_tools(tools)
    agent = {
        "input": lambda x: x["input"],
        "agent_db_retriever": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
    } | prompt | llm_with_tools | OpenAIToolsAgentOutputParser()
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # user_question = "What is the part name and description marked as 9?"
    if image_path is not None:
        result = agent_executor.invoke({"input": f"{user_question}, this is the image : {image_path}"})
    else:
        result = agent_executor.invoke({"input": f"{user_question}"})
    result = result['output']
    return result


