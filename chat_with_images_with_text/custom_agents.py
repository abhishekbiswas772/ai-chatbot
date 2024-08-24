from langchain.tools import tool
import pandas as pd

@tool
def get_part_details_from_db(part_id):
    """
    Retrieve part details from the car engine parts CSV file by part ID.

    Args:
        part_id (int): The ID of the part to retrieve.

    Returns:
        tuple: A tuple containing the part name and description if found, 
               otherwise None and an error message.
    """
    df = pd.read_csv("./car_engine_parts.csv")
    part_row = df[df['id'] == part_id]
    if not part_row.empty:
        part_name = part_row['partname'].values[0]
        description = part_row['description'].values[0]
        return part_name, description
    else:
        return None, "Part ID not found."

