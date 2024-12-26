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

