# from django.conf import settings


# # Save data to a MongoDB collection
# def save_to_mongo(collection_name, data):
#     collection = settings.MONGO_DB[collection_name]
#     result = collection.insert_one(data)
#     return result.inserted_id


# # Retrieve all documents from a MongoDB collection
# def fetch_from_mongo(collection_name):
#     collection = settings.MONGO_DB[collection_name]
#     documents = collection.find()
#     return list(documents)
