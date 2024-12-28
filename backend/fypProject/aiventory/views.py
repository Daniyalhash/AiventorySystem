from itertools import product
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
import csv
from io import StringIO
from datetime import datetime, timedelta
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.exceptions import ValidationError  # Add this import for ValidationError
from rest_framework.decorators import api_view
from rest_framework.response import Response
from bson import ObjectId
from io import BytesIO
import pandas as pd
from datetime import datetime
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
# MongoDB Atlas connection
client = MongoClient("mongodb+srv://syeddaniyalhashmi123:test123@cluster0.dutvq.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client["FYP"]

# Predefined mappings for keywords
COLUMN_MAP = {
    'vendor': ['vendor', 'reliability','ReliabilityScore', 'DeliveryTime','DeliveryTime', "new vendors", 'manufacturer', 'supplier'],
    'product': [
        'productname', 'category', 'subcategory', 'stock',
        'reorder', 'cost', 'selling', 'price', 'barcode',
        'expiry', 'past', 'sales', 'timespan', 'quantity'
    ]
    
    ,'unclassified': []  # This will be dynamically filled

}

REQUIRED_COLUMNS = [
    'productname', 'category', 'subcategory', 'vendor',
    'stockquantity', 'reorderthreshold', 'costprice',
    'sellingprice', 'timespan', 'expirydate', 'pastsalesdata',
    'DeliveryTime', 'ReliabilityScore', 'Barcode'
]

# used
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

        # Read the dataset into memory
        file_bytes = dataset_file.read()
        df = pd.read_csv(BytesIO(file_bytes))  # Load into pandas dataframe
        df.columns = df.columns.str.strip()

    # Assign IDs to vendors and process vendor data
        vendor_columns = ["vendor","DeliveryTime", "ReliabilityScore"]  # The column we are interested in for vendor data
        vendor_data = df[vendor_columns].drop_duplicates().reset_index(drop=True)
        # Generate unique vendor IDs
        vendor_data['_id'] = vendor_data.apply(lambda x: ObjectId(), axis=1)
        vendor_mapping = dict(zip(vendor_data['vendor'], vendor_data['_id']))

    # we can define vednor in vendor_columns
       # Simulate Product Columns and Mapping to Vendor IDs
        product_columns = [
            "productname", "category", "subcategory", "vendor", "stockquantity", "sellingprice", "Barcode", 
            "expirydate", "pastsalesdata", "timespan",'reorderthreshold', 'costprice'
        ]
        # Process product data (with foreign key)
        
        product_data = df[product_columns].drop_duplicates().reset_index(drop=True)
        # Map Vendor IDs to Product Data
        product_data['vendor_id'] = df['vendor'].map(vendor_mapping)
        
        
         # Remove the 'vendor' column from the product data after mapping vendor IDs
        product_data.drop(columns=['vendor'], inplace=True)

 # Perform column classification
        classified_columns = {
            'vendor': [],
            'product': [],
            'unclassified': []
        }
      # Classify columns as per your predefined mappings
        for col in df.columns:
            if col in COLUMN_MAP['vendor']:
                classified_columns['vendor'].append(col)
            elif col in COLUMN_MAP['product']:
                classified_columns['product'].append(col)
            else:
                classified_columns['unclassified'].append(col)

        # Check column existence and report any missing required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            return Response({"error": f"Missing required columns: {', '.join(missing_columns)}"}, status=400)

        # Output the classification result for debugging
        print("Classified Columns:", classified_columns)

        # Check if 'DeliveryTime' is being mapped and classified correctly
        if 'DeliveryTime' in classified_columns['product'] or 'DeliveryTime' in classified_columns['vendor']:
            print("'DeliveryTime' is classified correctly.")
        else:
            print("'DeliveryTime' is not classified correctly. Please check column mapping.")

        # Handle missing vendor IDs in product data
        if product_data['vendor_id'].isnull().any():
            missing_products = product_data[product_data['vendor_id'].isnull()]
            return Response({
                "error": "Some products do not have a corresponding vendor.",
                "missing_products": missing_products.to_dict(orient='records')
            }, status=400)

        
        # start here
        dataset_id = ObjectId()
        
 # for product db
        product_document = {
            "_id": ObjectId(),  # Generate unique ID for dataset
            "user_id": ObjectId(user_id),
            "dataset_id": dataset_id,
            "products": product_data.to_dict(orient="records"),  # Product references
            "upload_date": datetime.utcnow().isoformat(),
        }    
        db["products"].insert_one(product_document)
        
       
 # for vendor db

        vendor_document = {
            "_id": ObjectId(),  # Generate unique ID for dataset
            "user_id": ObjectId(user_id),
            "dataset_id": dataset_id,
            "vendors": vendor_data.to_dict(orient="records"),  # Vendor references
            "upload_date": datetime.utcnow().isoformat(),
        }
        db["vendors"].insert_one(vendor_document)
      
        # 3. Create Dataset Document
        dataset_document = {
            "_id": dataset_id,
            "user_id": ObjectId(user_id),
            "filename": dataset_file.name,
            "vendor_id": vendor_document["_id"],  # Reference the vendor document
            "product_id": product_document["_id"],   # Product references # make this foreign key
            "upload_date": datetime.utcnow().isoformat(),
        }
        db["datasets"].insert_one(dataset_document)
        

        # Update user document to reference this dataset
        dataset_info = {
            "dataset_id": dataset_id,
            "filename": dataset_file.name,
            "upload_date": dataset_document["upload_date"],
            "status": "uploaded"
        }
        db["users"].update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"datasets": dataset_info}}
        )
  # Return success response
        return Response({
            "message": f"Dataset '{dataset_file.name}' uploaded successfully!",
            "dataset_id": str(dataset_document["_id"]),
            "message": "Dataset successfully processed.",
          
        })
        

    except ValidationError as e:
        return Response({"error": str(e)}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


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

@api_view(['GET'])
def get_vendor(request):
    try:
        # Get User ID from request parameters
        user_id = request.query_params.get("user_id")
        if not user_id:
            return Response({"error": "User ID is required."}, status=400)

        # Validate if user exists
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            return Response({"error": "User not found."}, status=404)

        # Fetch vendors associated with the user
        vendors = list(db["vendors"].find({"user_id": ObjectId(user_id)}))

        # Prepare vendor data for response
        vendor_list = []
        for vendor in vendors:
            vendor["_id"] = str(vendor["_id"])  # Convert ObjectId to string for JSON response
            vendor_list.append(vendor)

        return Response({"vendors": vendor_list}, status=200)

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
#for login/signup 

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
#for login/signup 

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

#for login/signup 

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

#for dashboard counts


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

# for dashboard page analysis
@api_view(['GET'])
def product_benchmark(request):
    try:
        # Get user_id and productname from the query parameters
        user_id = request.query_params.get('user_id')
        category = request.query_params.get('category', 'Shampoo')  # Default to 'Shampoo'
        target_product = request.query_params.get('product', None)  # Product to analyze

        if not user_id:
            return Response({"error": "User ID is required!"}, status=400)
   
        if not target_product:
            return Response({"error": "Product name is missing in the query parameters."}, status=400)
        # Fetch user details from MongoDB
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            return Response({"error": "User not found!"}, status=404)

        # Fetch the user's dataset
        user_datasets = user.get("datasets", [])
        if not user_datasets:
            return Response({"error": "No datasets found for this user!"}, status=404)

        # Fetch the first dataset
        dataset_id = user_datasets[0].get("dataset_id")
        dataset_record = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
        if not dataset_record or "data" not in dataset_record:
            return Response({"error": "Dataset not found!"}, status=404)

        # Convert dataset into a pandas DataFrame
        df = pd.DataFrame(dataset_record["data"])

        # Filter products based on the given category
        category_products = df[df["category"] == category]
        if category_products.empty:
            return Response({"error": f"No products found in the category '{category}'!"}, status=404)

        # Find the target product
        target_product_data = category_products[category_products["productname"] == target_product]
        if target_product_data.empty:
            return Response({"error": f"Product '{target_product}' not found!"}, status=404)

        # Fetch 3 competitors randomly
        competitors = category_products[category_products["productname"] != target_product].sample(n=3, random_state=42)

        # Combine target product and competitors
        analysis_df = pd.concat([target_product_data, competitors])

        # Calculate benchmarks
        benchmarks = calculate_product_benchmarks(analysis_df)

        return Response({"benchmarks": benchmarks})

    except Exception as e:
        return Response({"error": str(e)}, status=500)

# for dashboard page analysis(Helper function to calculate product benchmarks)
def calculate_product_benchmarks(df):
    benchmarks = []

    for _, row in df.iterrows():
        productname = row.get("productname")
        sellingprice = row.get("sellingprice")
        costprice = row.get("costprice")
        
        # Handle missing values
        sellingprice = sellingprice if not pd.isna(sellingprice) else 0
        costprice = costprice if not pd.isna(costprice) else 0

        # Calculate profit margin
        profitmargin = (sellingprice - costprice) / sellingprice * 100 if sellingprice else 0

        benchmarks.append({
            "productname": productname,
            "sellingprice": sellingprice,
            "profitmargin": profitmargin,
        })
    
    return benchmarks

#to show dataset in invenoty page

@api_view(['GET'])
def get_current_dataset(request):
    user_id = request.GET.get("user_id", None)

    if not user_id:
        return Response({"error": "Missing user_id"}, status=400)

    # Find the user's dataset from MongoDB
    dataset = db["datasets"].find_one({"user_id": ObjectId(user_id)})
    if not dataset:
        return Response({"error": "No dataset found for the user."}, status=404)

    try:
        # Fetch the actual dataset and format it similarly to the dummy data
        data = dataset.get("data", [])
        if not data:
            return Response({"error": "Dataset is empty."}, status=404)

        # Format the dataset as you need for the frontend
        formatted_data = {
            "data": data
        }

        # Return the dataset in the same structure as the dummy data
        return Response(formatted_data)

    except Exception as e:
        # Log the exception for debugging
        print(f"Error: {e}")
        return Response({"error": "Internal server error"}, status=500)

# for visulization to show in invenoty page
@api_view(['GET'])
def get_inventory_visuals(request):
    try:
        # Get the user_id from the query parameters
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "User ID is required!"}, status=400)

        # Fetch user datasets
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            return Response({"error": "User not found!"}, status=404)

        # Fetch the user's dataset
        user_datasets = user.get("datasets", [])
        if not user_datasets:
            return Response({"error": "No datasets found for this user!"}, status=404)

        # Fetch the first dataset
        dataset_id = user_datasets[0].get("dataset_id")
        dataset_record = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
        if not dataset_record or "data" not in dataset_record:
            return Response({"error": "Dataset not found!"}, status=404)

        # Convert dataset into a pandas DataFrame
        df = pd.DataFrame(dataset_record["data"])

        # 1. Category-wise Profit Margin
        df['profit_margin'] = (df['sellingprice'] - df['costprice']) / df['costprice'] * 100
        category_profit_margin = df.groupby('category')['profit_margin'].mean().reset_index()
        category_profit_margin = category_profit_margin.to_dict(orient='records')

        # 2. Category-wise Cost
        category_cost = df.groupby('category')['costprice'].sum().reset_index()
        category_cost = category_cost.to_dict(orient='records')

        # 3. Product-wise Profit Margin (horizontal bar chart)
        product_profit_margin = df[['productname', 'profit_margin']].sort_values(by='profit_margin', ascending=False).head(10)
        product_profit_margin = product_profit_margin.to_dict(orient='records')

        # 4. Comparison of Selling Price and Cost Price
        product_price_comparison = df[['productname', 'sellingprice', 'costprice']].head(10)
        product_price_comparison = product_price_comparison.to_dict(orient='records')

        # Return data for visualizations
        return Response({
            "category_profit_margin": category_profit_margin,
            "category_cost": category_cost,
            "product_profit_margin": product_profit_margin,
            "product_price_comparison": product_price_comparison,
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(['GET'])
def get_insights_visuals(request):
    try:
        # Get the user_id from the query parameters
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response({"error": "User ID is required!"}, status=400)

        # Fetch user datasets
        user = db["users"].find_one({"_id": ObjectId(user_id)})
        if not user:
            return Response({"error": "User not found!"}, status=404)

        # Fetch the user's dataset
        user_datasets = user.get("datasets", [])
        if not user_datasets:
            return Response({"error": "No datasets found for this user!"}, status=404)

        # Fetch the first dataset
        dataset_id = user_datasets[0].get("dataset_id")
        dataset_record = db["datasets"].find_one({"_id": ObjectId(dataset_id)})
        if not dataset_record or "data" not in dataset_record:
            return Response({"error": "Dataset not found!"}, status=404)

        # Convert dataset into a pandas DataFrame
        df = pd.DataFrame(dataset_record["data"])
         # 1. Product Benchmarking: Compare 3 products in the same category based on selling price and profit margin.
        selected_category = "Electronics"  # Example category (you can make this dynamic)
        benchmarking_products = df[df['category'] == selected_category].head(3)
        benchmarking_data = benchmarking_products.assign(
            profit_margin=benchmarking_products['sellingprice'] - benchmarking_products['costprice']
        )[["productname", "sellingprice", "profit_margin"]].to_dict(orient="records")

        # 2. Identify Low Stock Products
        low_stock_products = df[df['stockquantity'] < df['reorderthreshold']]
        low_stock_data = low_stock_products[["productname", "category", "New Vendors", "stockquantity"]].to_dict(orient="records")

        # 3. Best Vendor Suggestion
        vendor_scores = (
            df.groupby('vendor')
            .agg(
                avg_reliability=('ReliabilityScore', 'mean'),
                avg_delivery_time=('DeliveryTime', 'mean'),
            )
            .sort_values(by=['avg_reliability', 'avg_delivery_time'], ascending=[False, True])
            .head(3)
            .reset_index()
            .to_dict(orient="records")
        )

        return Response({
            "benchmarking_data": benchmarking_data,
            "low_stock_data": low_stock_data,
            "vendor_suggestions": vendor_scores,
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)

# 

# Path where the dataset files are stored
@api_view(['GET'])
def get_total_products(request):
    user_id = request.query_params.get('user_id')
    if not user_id:
        return Response({"error": "User ID is required!"}, status=400)

    try:
        # Query the `products` collection for all documents associated with this user_id
        product_documents = db["products"].find({"user_id": ObjectId(user_id)})
        if not product_documents:
            return Response({"error": "No products found for this user!"}, status=404)

        # Initialize counters and sets
        total_unique_products = set()
        low_stock_products = 0
        total_vendors = set()

        # Low stock threshold
        low_stock_threshold = 5

        for product_doc in product_documents:
            # Access the `products` array inside each document
            products_array = product_doc.get("products", [])

            for product in products_array:
                # Extract product name and add to unique set
                product_name = product.get("productname")
                if product_name:
                    total_unique_products.add(product_name)

                # Check stock quantity for low stock
                stock_quantity = product.get("stockquantity", 0)
                if stock_quantity < low_stock_threshold:
                    low_stock_products += 1

                # Extract vendor_id and add to unique vendor set
                vendor_id = product.get("vendor_id")
                if vendor_id:
                    total_vendors.add(str(vendor_id))

        # Calculate totals
        total_unique_products_count = len(total_unique_products)
        total_vendors_count = len(total_vendors)
        print(total_unique_products_count)
        print(low_stock_products)
        print(total_vendors_count)
        # Return results
        return Response({
            "total_unique_products": total_unique_products_count,
            "low_stock_products": low_stock_products,
            "total_vendors": total_vendors_count
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)
# Fetch products, filter by category and user_id (optional)
@api_view(['GET'])
def get_products(request):
    """ Fetch all products or filter by category and user_id """
    category = request.GET.get('category', None)  # Get the category filter if provided
    user_id = request.GET.get('user_id', None)  # Get the user_id filter if provided
    
    query = {}
    if category:
        query['category'] = category
    if user_id:
        query['user_id'] = ObjectId(user_id)  # Filter by user_id

    # Query to get products from MongoDB
    products = list(products_collection.find(query))
    
    # Transform the MongoDB documents to a suitable format for the response
    products = [
        {
            "id": str(product["_id"]),
            "name": product["name"],
            "category": product["category"],
            "price": product["price"],
            "profitMargin": product["profitMargin"],
            "vendorId": str(product["vendorId"]),
        }
        for product in products
    ]
    
    return JsonResponse({"products": products})

# Fetch unique categories from the product dataset / done
@api_view(['GET'])
def get_categories(request):
    user_id = request.query_params.get('user_id')
    if not user_id:
        return Response({"error": "User ID is required!"}, status=400)

    try:
        # Query the `products` collection for all documents associated with this user_id
        product_documents = db["products"].find({"user_id": ObjectId(user_id)})
        if not product_documents:
            return Response({"error": "No products found for this user!"}, status=404)

        # Initialize a set to hold unique categories
        unique_categories = set()

        for product_doc in product_documents:
            # Access the `products` array inside each document
            products_array = product_doc.get("products", [])

            for product in products_array:
                # Extract category and add to unique categories set
                category = product.get("category")
                if category:
                    unique_categories.add(category)

        # Return the list of unique categories
        return Response({
            "categories": list(unique_categories)
        })

    except Exception as e:
        return Response({"error": str(e)}, status=500)


def convert_objectid(data):
    if isinstance(data, dict):
        return {key: convert_objectid(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data

@api_view(['GET'])
def get_top_products_by_category(request):
    user_id = request.query_params.get('user_id')
    category = request.query_params.get('category')
    
    if not user_id:
        return Response({"error": "User ID is required!"}, status=400)
    
    if not category:
        return Response({"error": "Category is required!"}, status=400)

    try:
        # Validate user_id
        try:
            user_id = ObjectId(user_id)
        except Exception:
            return Response({"error": "Invalid User ID format!"}, status=400)

        # Query the `products` collection
        product_documents = list(db["products"].find({"user_id": user_id}))
        if len(product_documents) == 0:
            return Response({"error": "No products found for this user!"}, status=404)

        # Initialize an array to hold products of the selected category
        products_in_category = []

        for product_doc in product_documents:
            # Access the `products` array
            products_array = product_doc.get("products", [])
            if not isinstance(products_array, list):
                continue  # Skip invalid products field

            for product in products_array:
                # Debug log for products
                print(f"Checking product: {product}")

                # Only add products that match the selected category
                if product.get("category") == category:
                    products_in_category.append({
                        "productname": product.get("productname"),
                        "category": product.get("category"),
                        "stockquantity": product.get("stockquantity"),
                        "sellingprice": product.get("sellingprice"),
                        "Barcode": product.get("Barcode"),
                        "expirydate": product.get("expirydate"),
                        "reorderthreshold": product.get("reorderthreshold"),
                        "costprice": product.get("costprice"),
                        "id": str(product.get("_id", "")),  # Default to empty string if _id is missing
                        "vendor_id": product.get("vendor_id")  # Ensure vendor_id exists
                    })

        # Return the top 5 products in the selected category
        top_5_products = products_in_category[:5]
        converted_data = convert_objectid(top_5_products)

        return Response({
            "products": converted_data
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print stack trace for debugging
        return Response({"error": str(e)}, status=500)



def calculate_profit_margin(selling_price, cost_price):
    if selling_price == 0:
        return 0
    return ((selling_price - cost_price) / selling_price) * 100



@api_view(['GET'])
def get_products_by_name(request):
    user_id = request.query_params.get('user_id')
    category = request.query_params.get('category')
    vendor_id = request.query_params.get('vendor_id')

    if not user_id:
        return Response({"error": "User ID is required!"}, status=400)
    
    if not category:
        return Response({"error": "Category is required!"}, status=400)
    if not vendor_id:
        return Response({"error": "vendor_id is required!"}, status=400)
    try:
        # Validate user_id
        try:
            user_id = ObjectId(user_id)
        except Exception:
            return Response({"error": "Invalid User ID format!"}, status=400)

        # Query the `products` collection
        product_documents = list(db["products"].find({"user_id": user_id}))
        if len(product_documents) == 0:
            return Response({"error": "No products found for this user!"}, status=404)
         # Query the `vendors` collection for the given user_id
        vendors = list(db["vendors"].find({"user_id": user_id}))
        if len(vendors) == 0:
            return Response({"error": "No vendors found for this user!"}, status=404)

        # Initialize an array to hold products of the selected category
        products_in_category = []
        vendor_Save = []
        vendorName = None  # Default value if no matching vendor is found
        DeliveryTime = None
        ReliabilityScore = None
        # Search for the vendor name
        for vendor_doc in vendors:
            vendor_array = vendor_doc.get("vendors", [])
            if not isinstance(vendor_array, list):
                continue  # Skip invalid vendors field

            for vendor in vendor_array:
                if str(vendor.get("_id")) == vendor_id:  # Ensure IDs are compared as strings
                    vendorName = vendor.get("vendor", "Unknown Vendor")
                    DeliveryTime = vendor.get("DeliveryTime", "Unknown DeliveryTime")
                    ReliabilityScore = vendor.get("ReliabilityScore", "Unknown ReliabilityScore")
                    break  # Stop searching once the vendor is found

            if vendorName:
                break  # Exit outer loop if the vendor is found

        if not vendorName:
            return Response({"error": "Vendor not found!"}, status=404)
                            
        for product_doc in product_documents:
            # Access the `products` array
            products_array = product_doc.get("products", [])
            if not isinstance(products_array, list):
                continue  # Skip invalid products field

            for product in products_array:
                # Only add products that match the selected category
                if product.get("category") == category:
                    selling_price = product.get("sellingprice", 0)
                    cost_price = product.get("costprice", 0)
                    profitmargin = calculate_profit_margin(selling_price, cost_price)
                
                    products_in_category.append({
                        "productname": product.get("productname"),
                        "category": product.get("category"),
                        "stockquantity": product.get("stockquantity"),
                        "sellingprice": selling_price,
                        "Barcode": product.get("Barcode"),
                        "expirydate": product.get("expirydate"),
                        "reorderthreshold": product.get("reorderthreshold"),
                        "costprice": cost_price,
                        "profitmargin": profitmargin,
                        "id": str(product.get("_id", "")),
                        "vendor": vendorName,
                        "DeliveryTime": DeliveryTime,
                        "ReliabilityScore": ReliabilityScore 
                    })

        # Get the top 5 products
        top_5_products = products_in_category[:5]

        # Convert ObjectIds to strings for serialization
        converted_data = convert_objectid(top_5_products)
        print(top_5_products)
        print("vendor found",vendorName)
        return Response({
            "products": converted_data
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print stack trace for debugging
        return Response({"error": str(e)}, status=500)
#error
@api_view(['GET'])
def get_vendor_by_name(request):
    user_id = request.query_params.get('user_id')
    
    if not user_id:
        return Response({"error": "User ID is required!"}, status=400)
    
    

    try:
        # Validate user_id
        try:
            user_id = ObjectId(user_id)
        except Exception:
            return Response({"error": "Invalid User ID format!"}, status=400)

 
         # Query the `vendors` collection for the given user_id
        vendors_documents = list(db["vendors"].find({"user_id": user_id}))
        if len(vendors_documents) == 0:
            return Response({"error": "No vendors found for this user!"}, status=404)

        # Initialize an array to hold products of the selected category
        vendor_found= []

        for vendor_doc in vendors_documents:
            # Access the `products` array
            vendor_found = vendor_doc.get("vendors", [])
            if not isinstance(vendor_found, list):
                continue  # Skip invalid products field

            for vendor in vendor_found:
                # Only add products that match the selected category

                    vendor_found.append({
            
                        "vendor": vendor.get("vendor") ,
                        "DeliveryTime": vendor.get("DeliveryTime"), 
                        "ReliabilityScore": vendor.get("ReliabilityScore") 
                    
                    })

        # Get the top 5 products
        top_5_vendor = vendor_found[:5]

        # Convert ObjectIds to strings for serialization
        converted_data = convert_objectid(top_5_vendor)
        print(top_5_vendor)
        # Example usage
        # userid = "676ec0b6165c0a9d91429edd"  
        # vendorid = "676ec0bc165c0a9d91429ede"
        # result = get_vendor_now(user_id, vendor_id)
        # print(result)  
        return Response({
            "products": converted_data
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print stack trace for debugging
        return Response({"error": str(e)}, status=500) 

#imp
# @api_view(['GET'])
# def get_products_by_name(request):
#     user_id = request.query_params.get('user_id')
#     category = request.query_params.get('category')
#     vendor_id = request.query_params.get('vendor_id')

#     if not user_id:
#         return Response({"error": "User ID is required!"}, status=400)
    
#     if not category:
#         return Response({"error": "Category is required!"}, status=400)
#     if not vendor_id:
#         return Response({"error": "vendor_id is required!"}, status=400)
#     try:
#         # Validate user_id
#         try:
#             user_id = ObjectId(user_id)
#         except Exception:
#             return Response({"error": "Invalid User ID format!"}, status=400)

#         # Query the `products` collection
#         product_documents = list(db["products"].find({"user_id": user_id}))
#         if len(product_documents) == 0:
#             return Response({"error": "No products found for this user!"}, status=404)
#          # Query the `vendors` collection for the given user_id
#         vendors = list(db["vendors"].find({"user_id": user_id}))
#         if len(vendors) == 0:
#             return Response({"error": "No vendors found for this user!"}, status=404)

#         # Initialize an array to hold products of the selected category
#         products_in_category = []

#         for product_doc in product_documents:
#             # Access the `products` array
#             products_array = product_doc.get("products", [])
#             if not isinstance(products_array, list):
#                 continue  # Skip invalid products field

#             for product in products_array:
#                 # Only add products that match the selected category
#                 if product.get("category") == category:
#                     selling_price = product.get("sellingprice", 0)
#                     cost_price = product.get("costprice", 0)
#                     profitmargin = calculate_profit_margin(selling_price, cost_price)
                
#                     products_in_category.append({
#                         "productname": product.get("productname"),
#                         "category": product.get("category"),
#                         "stockquantity": product.get("stockquantity"),
#                         "sellingprice": selling_price,
#                         "Barcode": product.get("Barcode"),
#                         "expirydate": product.get("expirydate"),
#                         "reorderthreshold": product.get("reorderthreshold"),
#                         "costprice": cost_price,
#                         "profitmargin": profitmargin,
#                        "id": str(product.get("_id", "")),
                       
#                     })

#         # Get the top 5 products
#         top_5_products = products_in_category[:5]

#         # Convert ObjectIds to strings for serialization
#         converted_data = convert_objectid(top_5_products)
#         print(top_5_products)
     
#         return Response({
#             "products": converted_data
#         })

#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())  # Print stack trace for debugging
#         return Response({"error": str(e)}, status=500)

def get_vendor_now(userid, vendorid):
    try:
        # Validate if user exists
        print(f"Looking for user with ID: {userid}")  # Debugging
        user = db.users.find_one({"_id": ObjectId(userid)})
        if not user:
            return {"error": "User not found."}
        print(f"User found: {user}")  # Debugging

        # Fetch vendor by vendor_id and user_id from the vendors collection
        print(f"Looking for vendor with ID: {vendorid} and user ID: {userid}")  # Debugging
        vendor = db.vendors.find_one(
            {"_id": ObjectId(vendorid), "user_id": ObjectId(userid)}
        )

        if not vendor:
            # Check if the vendor exists without matching user_id (for debugging purposes)
            print("Trying without matching user_id")  # Debugging
            vendor = db.vendors.find_one({"_id": ObjectId(vendorid)})
            if not vendor:
                return {"error": "Vendor not found."}
            return {"error": f"Vendor found, but user_id does not match. Vendor user_id: {vendor['user_id']}"}

        # Convert ObjectId to string for JSON-like response
        vendor["_id"] = str(vendor["_id"])
        vendor["user_id"] = str(vendor["user_id"])

        return {"vendor": vendor}

    except Exception as e:
        return {"error": str(e)}
# # Mock database
# db3 = {
#     "users1": [
#         {"_id": ObjectId("676ec0bc165c0a9d91429ede")},
#     ],
#     "vendors1": [
#         {"_id": ObjectId("64c8efb2a45cd32598f3b5f7"), "user_id": ObjectId("676ec0bc165c0a9d91429ede"), "name": "Vendor A"},
#         {"_id": ObjectId("64c8efb2a45cd32598f3b5f8"), "user_id": ObjectId("676ec0bc165c0a9d91429ede"), "name": "Vendor B"},
#     ]
# }

# def get_vendor_now(userid, vendorid):
#     try:
#         # Validate if user exists
#         user = next((u for u in db3["users1"] if u["_id"] == ObjectId(userid)), None)
#         if not user:
#             return {"error": "User not found."}

#         # Fetch vendor by vendor_id and user_id
#         vendor = next(
#             (v for v in db3["vendors1"] if v["_id"] == ObjectId(vendorid) and v["user_id"] == ObjectId(userid)),
#             None
#         )

#         if not vendor:
#             return {"error": "Vendor not found."}

#         # Convert ObjectId to string for JSON-like response
#         vendor["_id"] = str(vendor["_id"])
#         vendor["user_id"] = str(vendor["user_id"])

#         return {"vendor": vendor}

#     except Exception as e:
#         return {"error": str(e)}



# @api_view(['GET'])
# def get_products_by_name(request):
#     user_id = request.query_params.get('user_id')
#     category = request.query_params.get('category')
    
#     if not user_id:
#         return Response({"error": "User ID is required!"}, status=400)
    
#     if not category:
#         return Response({"error": "Category is required!"}, status=400)

#     try:
#         # Validate user_id
#         try:
#             user_id = ObjectId(user_id)
#         except Exception:
#             return Response({"error": "Invalid User ID format!"}, status=400)

#         # Query the `products` collection
#         product_documents = list(db["products"].find({"user_id": user_id}))
#         if len(product_documents) == 0:
#             return Response({"error": "No products found for this user!"}, status=404)
#          # Query the `vendors` collection for the given user_id
#         vendors = list(db["vendors"].find({"user_id": user_id}))
#         if len(vendors) == 0:
#             return Response({"error": "No vendors found for this user!"}, status=404)

#         # Initialize an array to hold products of the selected category
#         products_in_category = []

#         for product_doc in product_documents:
#             # Access the `products` array
#             products_array = product_doc.get("products", [])
#             if not isinstance(products_array, list):
#                 continue  # Skip invalid products field

#             for product in products_array:
#                 # Only add products that match the selected category
#                 if product.get("category") == category:
#                     selling_price = product.get("sellingprice", 0)
#                     cost_price = product.get("costprice", 0)
#                     profitmargin = calculate_profit_margin(selling_price, cost_price)
                    
#                     # Find vendor info
#                     vendor_id = product.get("vendor_id")
#                     vendor_info = None
#                     if vendor_id:
#                         # Check if the vendor_id matches any vendor in the user's vendors
#                         vendor_info = next(
#                             (vendor for vendor in vendors if str(vendor["_id"]) == str(vendor_id)),
#                             None
#                         )

#                     products_in_category.append({
#                         "productname": product.get("productname"),
#                         "category": product.get("category"),
#                         "stockquantity": product.get("stockquantity"),
#                         "sellingprice": selling_price,
#                         "Barcode": product.get("Barcode"),
#                         "expirydate": product.get("expirydate"),
#                         "reorderthreshold": product.get("reorderthreshold"),
#                         "costprice": cost_price,
#                         "profitmargin": profitmargin,
#                        "id": str(product.get("_id", "")),
#                         "vendor_info": {
#                             "vendor": vendor_info.get("vendor") if vendor_info else None,
#                             "DeliveryTime": vendor_info.get("DeliveryTime") if vendor_info else None,
#                             "ReliabilityScore": vendor_info.get("ReliabilityScore") if vendor_info else None,
#                         }
#                     })

#         # Get the top 5 products
#         top_5_products = products_in_category[:5]

#         # Convert ObjectIds to strings for serialization
#         converted_data = convert_objectid(top_5_products)
#         print(top_5_products)
#         userid = "676ec0bc165c0a9d91429ede"
#         vendorid = "64c8efb2a45cd32598f3b5f7"
#         result = get_vendor_now(userid, vendorid)
#         print(result)   
#         return Response({
#             "products": converted_data
#         })

#     except Exception as e:
#         import traceback
#         print(traceback.format_exc())  # Print stack trace for debugging
#         return Response({"error": str(e)}, status=500)
@api_view(['GET'])
def get_vendor_by_id(request):
    try:
        # Fetch Vendor ID from request parameters
        vendor_id = request.query_params.get("vendor_id")
        if not vendor_id:
            return Response({"error": "Vendor ID is required."}, status=400)

        # Validate if Vendor ID is in proper ObjectId format
        try:
            vendor_object_id = ObjectId(vendor_id)
        except Exception:
            return Response({"error": "Invalid Vendor ID format."}, status=400)

        # Access the vendor from the database
        vendor = db["vendors"].find_one({"_id": vendor_object_id})
        if not vendor:
            return Response({"error": "Vendor not found."}, status=404)

        # Prepare and return the vendor data
        vendor["_id"] = str(vendor["_id"])  # Convert ObjectId to string for JSON response
        return Response({"vendor": vendor}, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)