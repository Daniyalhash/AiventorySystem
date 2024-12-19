from pymongo import MongoClient
from bson import ObjectId
import pandas as pd
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Function to calculate lengths from DataFrame
def calculate_lengths(df):
    """Calculate lengths of products, low stock products, and total vendors."""
    try:
        total_products_length = df['productname'].nunique()
        low_stock_length = df[df['stockquantity'] < df['reorderthreshold']].shape[0]
        total_vendors_length = df['New Vendors'].nunique()
        print(f"Total unique products: {total_products_length}")
        print(f"Total low stock products: {low_stock_length}")
        print(f"Total unique vendors: {total_vendors_length}")
        return total_products_length, low_stock_length, total_vendors_length
    except Exception as e:
        print(f"Error in calculate_lengths: {str(e)}")
        raise e  # Propagate the exception to handle it at a higher level

# # Function to analyze datasets for a specific user
# def analyze_lengths(db, user_id):
#     try:
#         print(f"Analyzing datasets for user_id: {user_id}")  # Debug: Check received user_id

#         # Fetch user's datasets
#         user = db["users"].find_one({"_id": ObjectId(user_id)})
#         if not user or 'datasets' not in user:
#             print("No datasets found for the user.")  # Debug
#             return {"error": "No datasets found for the user."}

#         # Initialize counters
#         total_products = set()
#         total_vendors = set()
#         low_stock_count = 0

#         for dataset_info in user["datasets"]:
#             print(f"Processing dataset: {dataset_info}")  # Debug

#             # Fetch dataset from the database
#             dataset = db["datasets"].find_one({"filename": dataset_info["filename"]})
#             if not dataset:
#                 print(f"Dataset not found: {dataset_info['filename']}")  # Debug
#                 continue

#             # Debug: Display fetched dataset
#             print(f"Fetched dataset: {dataset}")

#             # Update counts
#             total_products.update(dataset.get("productname", []))
#             total_vendors.update(dataset.get("New Vendors", []))
#             low_stock_products = [
#                 product for product in dataset.get("products", [])
#                 if product["stockquantity"] < product["reorderthreshold"]
#             ]
#             low_stock_count += len(low_stock_products)

#             # Debug: Print counts after processing dataset
#             print(f"Total products so far: {len(total_products)}")
#             print(f"Total vendors so far: {len(total_vendors)}")
#             print(f"Low stock products so far: {low_stock_count}")

#         # Final stats
#         stats = {
#             "total_products": len(total_products),
#             "total_vendors": len(total_vendors),
#             "low_stock": low_stock_count,
#         }

#         print(f"Final stats: {stats}")  # Debug
#         return stats

#     except Exception as e:
#         print(f"Error during analysis: {str(e)}")  # Debug
#         return {"error": str(e)}
