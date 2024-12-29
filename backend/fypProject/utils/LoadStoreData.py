from bson import ObjectId
from datetime import datetime
import pandas as pd
from io import BytesIO

class LoadStoreData:
    def __init__(self, db):
        self.db = db

    def read_dataset(self, file):
        file_bytes = file.read()
        df = pd.read_csv(BytesIO(file_bytes))
        df.columns = df.columns.str.strip()
        return df

    def classify_columns(self, df, column_map):
        classified_columns = {'vendor': [], 'product': [], 'unclassified': []}
        for col in df.columns:
            if col in column_map['vendor']:
                classified_columns['vendor'].append(col)
            elif col in column_map['product']:
                classified_columns['product'].append(col)
            else:
                classified_columns['unclassified'].append(col)
        return classified_columns

    def process_vendor_data(self, df, vendor_columns):
        vendor_data = df[vendor_columns].drop_duplicates().reset_index(drop=True)
        vendor_data['_id'] = vendor_data.apply(lambda x: ObjectId(), axis=1)
        vendor_mapping = dict(zip(vendor_data['vendor'], vendor_data['_id']))
        return vendor_data, vendor_mapping

    def process_product_data(self, df, product_columns, vendor_mapping):
        product_data = df[product_columns].drop_duplicates().reset_index(drop=True)
        product_data['vendor_id'] = df['vendor'].map(vendor_mapping)
        product_data.drop(columns=['vendor'], inplace=True)
        return product_data

    def save_vendor_data(self, user_id, dataset_id, vendor_data):
        vendor_document = {
            "_id": ObjectId(),
            "user_id": ObjectId(user_id),
            "dataset_id": dataset_id,
            "vendors": vendor_data.to_dict(orient="records"),
            "upload_date": datetime.utcnow().isoformat(),
        }
        self.db["vendors"].insert_one(vendor_document)
        return vendor_document

    def save_product_data(self, user_id, dataset_id, product_data):
        product_document = {
            "_id": ObjectId(),
            "user_id": ObjectId(user_id),
            "dataset_id": dataset_id,
            "products": product_data.to_dict(orient="records"),
            "upload_date": datetime.utcnow().isoformat(),
        }
        self.db["products"].insert_one(product_document)
        return product_document

    def save_dataset_data(self, user_id, dataset_id, dataset_file_name, vendor_doc_id, product_doc_id):
        dataset_document = {
            "_id": dataset_id,
            "user_id": ObjectId(user_id),
            "filename": dataset_file_name,
            "vendor_id": vendor_doc_id,
            "product_id": product_doc_id,
            "upload_date": datetime.utcnow().isoformat(),
        }
        self.db["datasets"].insert_one(dataset_document)
        return dataset_document

    def update_user_dataset(self, user_id, dataset_info):
        self.db["users"].update_one(
            {"_id": ObjectId(user_id)},
            {"$push": {"datasets": dataset_info}}
        )
