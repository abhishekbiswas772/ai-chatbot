from pymilvus import connections, utility, CollectionSchema, DataType, FieldSchema, Collection
import numpy as np
import os


DB_USER = "minioadmin"
DB_PASSWORD = "minioadmin"
DB_HOST = "localhost"
DB_PORT = "19530"
DB_COLLECTION_NAME = "partEmbeddingEngine"

class MilvisHandler:
    @staticmethod
    def connect_to_milvus():
        try:
            connections.connect("default",user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            print("Connected to Milvus.")
            return 1
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    @staticmethod
    def create_collection(name, fields, description, consistency_level="Strong"):
        try:
            schema = CollectionSchema(fields, description)
            collection = Collection(name, schema, consistency_level=consistency_level)
            print(f"Collection '{name}' created.")
            return collection
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return None
        

    @staticmethod
    def insert_data(collection, entity):
        try:
            data = [
                [entity['pid']], 
                [entity['vector']]
            ]
            collection.insert(data)
            collection.flush()
            print(f"Inserted data into '{collection.name}'. Number of entities: {collection.num_entities}")

            return {"collection_name" : collection.name, "collection_num_entities" : collection.num_entities}

        except Exception as e:
            print(f"Failed to insert data: {e}")
            return None


    @staticmethod
    def create_index(collection, field_name, index_type, metric_type, params):
        try:
            index = {"index_type": index_type, "metric_type": metric_type, "params": params}
            collection.create_index(field_name, index)
            print(f"Index '{index_type}' created for field '{field_name}'.")
        except Exception as e:
            print(f"Failed to create index: {e}")


    @staticmethod
    def create_collection(name, fields, description, consistency_level="Strong"):
        try:
            schema = CollectionSchema(fields, description)
            collection = Collection(name, schema, consistency_level=consistency_level)
            print(f"Collection '{name}' created.")
            return collection
        except Exception as e:
            print(f"Failed to create collection: {e}")
            return None

    @staticmethod
    def insert_into_db(entity, dims):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=90, is_primary=True, auto_id=True),
            FieldSchema(name="pid", dtype=DataType.VARCHAR, max_length=90),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim= dims)
        ]
        collection = MilvisHandler.create_collection(DB_COLLECTION_NAME, fields, "Collection for Dev Milvus")

        if collection is not None:
            result = MilvisHandler.insert_data(collection, entity)
            print(">>> result == ",result)
            return [1, result]
        else:
            print("Collection creation failed. Aborting further operations.")
            return [0,"Collection creation failed"]


    @staticmethod
    def search_and_query(collection, query_embeddings):
        try:
            # Perform the search
            res = collection.search(
                data=query_embeddings,
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {}},
                limit=1,
                expr=None,
                output_fields=["pid"]  # Ensure 'pid' is retrieved in search results
                
            )
            # print(res)

            filtered_results = []
            for hits in res:
                for hit in hits:
                    entity_pid = hit.entity.get('pid')
                    filtered_results.append({
                        "id": hit.id,
                        "pid": entity_pid
                    })

            if filtered_results:
                return filtered_results
            else:
                print("No results matching beyond 80%")
                return ["No results matching beyond 80%"]
        except Exception as e:
            print(f"Search failed: {e}")
            return ["Search failed due to an error"]
        

    @staticmethod
    def drop_collection(collection_name):
        try:
            connections.connect("default",user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
            print("Connected to Milvus.")
            utility.drop_collection(collection_name)
            print(f"Dropped collection '{collection_name}'.")
            return 1
        except Exception as e:
            print(f"Failed to drop collection: {e}")



    @staticmethod
    def semantic_search(entity):
        try:
            collection = Collection(name=DB_COLLECTION_NAME)
            MilvisHandler.create_index(collection, "vector", "IVF_FLAT", "COSINE", {"nlist": 128})
            collection.load()
            print(f"Collection '{collection.name}' loaded successfully.")

            # Ensure the vector is a 2D array for Milvus search
            query_embeddings = np.array(entity["vector"], dtype=np.float32).reshape(1, -1)
            result = MilvisHandler.search_and_query(collection, query_embeddings)
            return [1, result]
        except Exception as e:
            print(f"Failed to load collection or perform search: {e}")
            return [0, str(e)]

