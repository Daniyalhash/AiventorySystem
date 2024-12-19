from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from pymongo import MongoClient
import bcrypt
from utils.csv_validator import validate_columns
from utils.validate_file_extension import validate_file_extension
import pandas as pd
import jwt
import os
from django.core.files.storage import FileSystemStorage
from bson import ObjectId  # Import ObjectId from bson to handle MongoDB IDs

from datetime import datetime, timedelta
from io import BytesIO

# MongoDB Atlas connection
client = MongoClient("mongodb+srv://syeddaniyalhashmi123:test123@cluster0.dutvq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["FYP"]



# Example usage
required_columns = [
    'productname', 'category', 'subcategory', 'vendor', 
    'stockquantity', 'reorderthreshold', 'costprice', 
    'sellingprice', 'timespan', 'expirydate', 'pastsalesdata',
    'DeliveryTime', 'ReliabilityScore', 'Barcode', 
    'DeliveryTime_Normalized', 'New Vendors'
]


# for signUp done
@api_view(['POST'])
def signup(request):
    try:
        data = request.data
        if db["users"].find_one({"email": data["email"]}):
            return Response({"error": "User already exists!"}, status=400)

        hashed_password = bcrypt.hashpw(data["password"].encode('utf-8'), bcrypt.gensalt())
        user = {
            "username": data["username"],
            "email": data["email"],
            "phone": data["phone"],
            "shop_name": data["shop_name"],
            "password": hashed_password.decode('utf-8'),
            "status": "incomplete",  # New field
        }
        result = db["users"].insert_one(user)
        user_id = str(result.inserted_id)  # Get the user_id (MongoDB ObjectId as a string)

        return Response({"message": "User registered successfully!", "user_id": user_id})  # Return user_id in response
    except Exception as e:
        return Response({"error": str(e)}, status=500)

# dataset slightly done
# data save horaha ha file ma kyu?
# user has option to upload only one dataset
@api_view(['POST'])
def upload_dataset(request):
    try:
        # Get User ID
        user_id = request.data.get("user_id")
        if not user_id:
            return Response({"error": "User ID is required."}, status=400)

        # Check if the user already has a dataset uploaded
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if user and user.get("datasets") and len(user["datasets"]) > 0:
            return Response({"error": "You can only upload one dataset."}, status=400) 
        # Check Dataset File
        if 'dataset' not in request.FILES:
            return Response({"error": "Dataset file is required."}, status=400)

        # Get Dataset File
        dataset_file = request.FILES['dataset']
        
        # Validate file type and size
        validate_file_extension(dataset_file.name)
        # validate_file_size(dataset_file)

        # Read the dataset into memory without saving it to the file system
        file_bytes = dataset_file.read()
        df = pd.read_csv(BytesIO(file_bytes))  # Load into pandas dataframe

        # Validate Columns (Assuming `required_columns` is defined)
        validation_result = validate_columns(df, required_columns)
        if "error" in validation_result:
            return Response(validation_result, status=400)

        

        # Prepare Dataset Document
        dataset_data = df.to_dict(orient="records")
        dataset_document = {
            "_id": ObjectId(),  # Generate Dataset ID
            "user_id": ObjectId(user_id),
            "filename": dataset_file.name,
            "data": dataset_data,
            "upload_date": datetime.utcnow().isoformat(),
        }

        # Insert Dataset into 'datasets' Collection
        db["datasets"].insert_one(dataset_document)

        # Update User Document with Dataset Reference
        dataset_info = {
            "dataset_id": dataset_document["_id"],
            "filename": dataset_file.name,
            "upload_date": dataset_document["upload_date"],
            "status": "uploaded"
        }
        db["users"].update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"datasets": dataset_info}}
        )

        return Response({
            "message": f"Dataset '{dataset_file.name}' uploaded successfully!",
            "dataset_id": str(dataset_document["_id"])
        })

    except ValidationError as e:
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
# get details  done
@api_view(['GET'])
def get_user_details(request):
    try:
        user_id = request.query_params.get('user_id')  # Get user_id from query params
        if not user_id:
            return Response({"error": "User ID is required!"}, status=400)

        # Convert the user_id to an ObjectId
        user_id = ObjectId(user_id)
        
        # Query the database for the user by ID
        user = db["users"].find_one({"_id": user_id})

        if not user:
            return Response({"error": "User not found!"}, status=404)

        # Return user details
        user_details = {
            "username": user["username"],
            "email": user["email"],
            "phone": user.get("phone", "N/A"),  # Default to "N/A" if no phone exists
        }

        return Response(user_details)

    except Exception as e:
        return Response({"error": f"Error fetching user details: {str(e)}"}, status=500)
    
    
# Create another endpoint to mark the user as "complete" once they press the dashboard button.

@api_view(['POST'])
def complete_signup(request):
    try:
        user_id = request.data.get("user_id")  # Get user_id from the request
        if not user_id:
            return Response({"error": "User ID is required."}, status=400)

        result = db["users"].update_one(
            {"_id": ObjectId(user_id), "status": "incomplete"},  # Query by user_id
            {"$set": {"status": "complete"}}  # Set status to "complete"
        )

        if result.matched_count == 0:
            return Response({"error": "No such user found or user already completed signup."}, status=404)

        return Response({"message": "User signup completed successfully."})

    except Exception as e:
        return Response({"error": str(e)}, status=500)


# delete a user based on cancel button while registeration
# delete a user based on user_id
# not use yet
@api_view(['POST'])
def delete_user(request):
    try:
        user_id = request.data.get("user_id")
        if not user_id or not ObjectId.is_valid(user_id):
            return Response({"error": "Invalid User ID."}, status=400)

        result = db["users"].delete_one({"_id": ObjectId(user_id), "status": "incomplete"})
        if result.deleted_count == 0:
            return Response({"error": "No such user found or user already completed signup."}, status=404)

        return Response({"message": "User data deleted successfully."})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

    

# for login donw
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "fallback_secret_key")

@api_view(['POST'])
def login(request):
    data = request.data
    user = db["users"].find_one({"email": data["email"]})

    if not user:
        return Response({"error": "User does not exist!"}, status=404)

    # Verify password
    if not bcrypt.checkpw(data["password"].encode('utf-8'), user["password"].encode('utf-8')):
        return Response({"error": "Invalid password!"}, status=400)

      # Generate JWT token
    payload = {
        "id": str(user["_id"]),
        "email": user["email"],
        "exp": datetime.utcnow() + timedelta(hours=24)  # Token valid for 24 hours
    
    }

    # Correct usage of jwt.encode() with PyJWT
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    return Response({
        "message": "Login successful!", 
        "token": token,
        "userId": str(user["_id"])  # Include userId in the response
    })

@api_view(['GET'])
def validate_token(request):
    token = request.headers.get("Authorization", "").split(" ")[1]  # Get token from the header
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user = db["users"].find_one({"_id": ObjectId(payload["id"])}, {"_id": 0, "password": 0})
        if not user:
            return Response({"error": "User not found."}, status=404)
        return Response({"user": user})
    except jwt.ExpiredSignatureError:
        return Response({"error": "Token has expired!"}, status=401)
    except jwt.InvalidTokenError:
        return Response({"error": "Invalid token!"}, status=401)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

# Path where the dataset files are stored
@api_view(['GET'])
def get_total_products(request):
    user_id = request.query_params.get('user_id')
    if not user_id:
        return Response({"error": "User ID is required!"}, status=400)

    try:
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            return Response({"error": "User not found!"}, status=404)

        # Find the user by user_id
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        
        if not user:
            return Response({"error": "User not found!"}, status=404)
        
        # Get all datasets linked to the user
        user_datasets = user.get("datasets", [])
        
        if not user_datasets:
            return Response({"error": "No datasets found for this user!"}, status=404)
        
        total_unique_products = set()  # To store unique product names
        low_stock_products = 0  # Counter for low stock products
        vendors = set()  # Set to store unique vendors  
         # Define the low stock threshold
        low_stock_threshold = 5  # Example threshold for low stock (can be adjusted)      
        for dataset in user_datasets:
            dataset_id = dataset.get("dataset_id")
            
            if not dataset_id:
                continue  # Skip datasets without an ID
            
            # Fetch the dataset from the datasets collection
            dataset_record = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
            
            if not dataset_record or "data" not in dataset_record:
                continue  # Skip if dataset is missing or malformed
            
            # Extract product data and process
            for item in dataset_record["data"]:
                productname = item.get("productname")
                stockquantity = item.get("stockquantity", 0)
                reorderthreshold = item.get("reorderthreshold", 0)
                vendor = item.get("New Vendors")
                
                if productname:
                    total_unique_products.add(productname)  # Add product name to the unique set
                
                if stockquantity < low_stock_threshold:
                    low_stock_products += 1  # Count low stock products
                
                if vendor:
                    vendors.add(vendor)  # Add vendor to the set
        
        # Calculate the total number of unique products, vendors, and low stock products
        total_count = len(total_unique_products)
        total_vendors = len(vendors)

        # Return the response with all the required details
        return Response({
            "total_unique_products": total_count,
            "low_stock_products": low_stock_products,
            "total_vendors": total_vendors
        })
    
    except Exception as e:
        return Response({"error": str(e)}, status=500)

# mongodb+srv://syeddaniyalhashmi123:test123@cluster0.dutvq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
@api_view(['GET'])
def suggest_vendors_for_user(request):
    try:
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "User ID is required!"}, status=400)

        # Fetch the user's dataset from MongoDB
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            return Response({"error": "User not found!"}, status=404)

        # Assuming user has a dataset linked
        user_dataset = user["datasets"][0]  # Get first dataset
        dataset_id = user_dataset["dataset_id"]
        dataset_record = db["datasets"].find_one({"_id": ObjectId(dataset_id)})

        if not dataset_record or "data" not in dataset_record:
            return Response({"error": "Dataset not found!"}, status=404)

        # Convert the dataset into a DataFrame
        df = pd.DataFrame(dataset_record["data"])

        # Create an instance of ProductVendorManager
        manager = ProductVendorManager(df)

        # Get the suggested vendors for low-stock products
        suggested_vendors = manager.suggest_vendors_for_low_stock_products()

        if suggested_vendors.empty:
            return Response({"message": "No low stock products found."})

        return Response(suggested_vendors.to_dict(orient="records"))

    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
    # @api_view(['POST'])

@api_view(['GET'])
def product_benchmark(request):
    try:
        # Get the user_id and dataset_id from the query parameters
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "User ID is required!"}, status=400)

        # Fetch user details from MongoDB
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            return Response({"error": "User not found!"}, status=404)

        # Fetch the user's dataset
        user_datasets = user.get("datasets", [])
        if not user_datasets:
            return Response({"error": "No datasets found for this user!"}, status=404)
        
        # Fetch the first dataset (you can modify this to handle multiple datasets)
        dataset_id = user_datasets[0].get("dataset_id")
        dataset_record = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
        if not dataset_record or "data" not in dataset_record:
            return Response({"error": "Dataset not found!"}, status=404)
        
        # Convert dataset into a pandas DataFrame
        df = pd.DataFrame(dataset_record["data"])

        # Calculate product benchmarks
        benchmarks = calculate_product_benchmarks(df)

        # Sort products by total benchmark score (or any other metric you want to prioritize)
        sorted_benchmarks = sorted(benchmarks, key=lambda x: x['total_benchmark'], reverse=True)

        # Select the top 3 benchmarks
        top_3_benchmarks = sorted_benchmarks[:3]

        # Get vendors associated with these products (assuming 'vendor' is a field in the dataset)
        # Assuming that a vendor is associated with each product in the dataset
        vendors = {vendor: [] for vendor in set([item['category'] for item in top_3_benchmarks])}

        # Aggregate top vendors for the selected top 3 products
        for product in top_3_benchmarks:
            vendor = product['category']  # Assuming 'category' is the vendor here
            vendors[vendor].append(product)

        # Get the top 2 vendors based on their product benchmarks
        top_vendors = sorted(vendors.items(), key=lambda x: sum([product['total_benchmark'] for product in x[1]]), reverse=True)[:2]

        # Prepare the result to return
        result = {
            "benchmarks": top_3_benchmarks,
            "top_vendors": top_vendors
        }

        # Return the benchmark results
        return Response(result)

    except Exception as e:
        return Response({"error": str(e)}, status=500)


# Helper function to calculate product benchmarks based on dynamic values
def calculate_product_benchmarks(df):
    benchmarks = []

    # Example: Dynamic benchmark based on product features (price, stock, reliability score, etc.)
    for _, row in df.iterrows():
        productname = row.get("productname")
        category = row.get("category")  # Assuming 'category' represents the vendor
        sellingprice = row.get("sellingprice")
        stockquantity = row.get("stockquantity")
        delivery_time = row.get("DeliveryTime")
        reliability_score = row.get("ReliabilityScore")
        
        # Check for missing values and set default if necessary
        if pd.isna(sellingprice):
            sellingprice = 0
        if pd.isna(stockquantity):
            stockquantity = 0
        if pd.isna(delivery_time):
            delivery_time = 0
        if pd.isna(reliability_score):
            reliability_score = 0

        # Example benchmark calculations (can be adjusted as needed)
        price_benchmark = sellingprice / 100  # Normalize price for benchmark
        stock_benchmark = 100 - (stockquantity / 100)  # Lower stock -> higher benchmark
        delivery_benchmark = 10 - (delivery_time / 2)  # Shorter delivery time -> higher benchmark
        reliability_benchmark = reliability_score * 10  # Higher reliability score -> higher benchmark

        # Aggregate the benchmarks into a single value (this can be adjusted based on your logic)
        total_benchmark = (price_benchmark + stock_benchmark + delivery_benchmark + reliability_benchmark) / 4

        # Append the product's benchmark data
        benchmarks.append({
            "productname": productname,
            "category": category,
            "price_benchmark": price_benchmark,
            "stock_benchmark": stock_benchmark,
            "delivery_benchmark": delivery_benchmark,
            "reliability_benchmark": reliability_benchmark,
            "total_benchmark": total_benchmark
        })
    
    return benchmarks
# def product_benchmark(request):
#     try:
#         # Get the user_id and dataset_id from the query parameters
#         user_id = request.query_params.get('user_id')
#         if not user_id:
#             return Response({"error": "User ID is required!"}, status=400)

#         # Fetch user details from MongoDB
#         user = db["users"].find_one({"_id": ObjectId(user_id)})
#         if not user:
#             return Response({"error": "User not found!"}, status=404)

#         # Fetch the user's dataset
#         user_datasets = user.get("datasets", [])
#         if not user_datasets:
#             return Response({"error": "No datasets found for this user!"}, status=404)
        
#         # Fetch the first dataset (you can modify this to handle multiple datasets)
#         dataset_id = user_datasets[0].get("dataset_id")
#         dataset_record = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
#         if not dataset_record or "data" not in dataset_record:
#             return Response({"error": "Dataset not found!"}, status=404)
        
#         # Convert dataset into a pandas DataFrame
#         df = pd.DataFrame(dataset_record["data"])

#         # Calculate product benchmarks
#         benchmarks = calculate_product_benchmarks(df)

#         # Return the benchmark results
#         return Response({"benchmarks": benchmarks})

#     except Exception as e:
#         return Response({"error": str(e)}, status=500)


# # Helper function to calculate product benchmarks based on dynamic values
# def calculate_product_benchmarks(df):
#     benchmarks = []

#     # Example: Dynamic benchmark based on product features (price, stock, reliability score, etc.)
#     for _, row in df.iterrows():
#         productname = row.get("productname")
#         category = row.get("category")
#         sellingprice = row.get("sellingprice")
#         stockquantity = row.get("stockquantity")
#         delivery_time = row.get("DeliveryTime")
#         reliability_score = row.get("ReliabilityScore")
        
#         # Check for missing values and set default if necessary
#         if pd.isna(sellingprice):
#             sellingprice = 0
#         if pd.isna(stockquantity):
#             stockquantity = 0
#         if pd.isna(delivery_time):
#             delivery_time = 0
#         if pd.isna(reliability_score):
#             reliability_score = 0

#         # Example benchmark calculations (can be adjusted as needed)
#         price_benchmark = sellingprice / 100  # Normalize price for benchmark
#         stock_benchmark = 100 - (stockquantity / 100)  # Lower stock -> higher benchmark
#         delivery_benchmark = 10 - (delivery_time / 2)  # Shorter delivery time -> higher benchmark
#         reliability_benchmark = reliability_score * 10  # Higher reliability score -> higher benchmark

#         # Aggregate the benchmarks into a single value (this can be adjusted based on your logic)
#         total_benchmark = (price_benchmark + stock_benchmark + delivery_benchmark + reliability_benchmark) / 4

#         # Append the product's benchmark data
#         benchmarks.append({
#             "productname": productname,
#             "category": category,
#             "price_benchmark": price_benchmark,
#             "stock_benchmark": stock_benchmark,
#             "delivery_benchmark": delivery_benchmark,
#             "reliability_benchmark": reliability_benchmark,
#             "total_benchmark": total_benchmark
#         })
    
#     return benchmarks