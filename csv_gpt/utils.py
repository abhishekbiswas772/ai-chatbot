import os
from typing import TextIO
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent

# from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI


class CSVProcessor:
    @staticmethod
    def load_openai_key():
        os.environ["OPEN_AI_KEY"] = (
            "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"
        )
        api_key = os.getenv("OPEN_AI_KEY")
        return api_key

    @staticmethod
    def get_answer_from_llm(file: TextIO, query: str) -> str:
        """
        Returns the answer to the given query by querying a CSV file.
        Args:
        - file (str): the file path to the CSV file to query.
        - query (str): the question to ask the agent.
        Returns:
        - answer (str): the answer to the query from the CSV file.
        """
        api_key = CSVProcessor.load_openai_key()
        agent = create_csv_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=api_key), file, verbose=True,
            allow_dangerous_code = True
        )
        answer = agent.run(query)
        return answer

