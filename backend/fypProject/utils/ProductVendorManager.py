import pandas as pd
# class ProductVendorManager:
#     def __init__(self, dataframe):
#         """Initialize the ProductVendorManager with the given DataFrame."""
#         self.df = dataframe

#     def suggest_vendors_for_low_stock_products(self):
#         """Suggest vendors for all products that are low on stock based on price, delivery time, and reliability score."""
#         # Identify low-stock products
#         low_stock_products = self.df[self.df['stockquantity'] < self.df['reorderthreshold']]

#         if low_stock_products.empty:
#             print("No products are low on stock.")
#             return pd.DataFrame()

#         # Calculate a ranking score for each vendor based on Price, DeliveryTime, and ReliabilityScore
#         self.df['RankScore'] = (
#             self.df['costprice'] * 0.5 +            # Price weight: 50%
#             self.df['DeliveryTime'] * 0.3 +        # Delivery time weight: 30%
#             (1 - self.df['ReliabilityScore']) * 0.2  # Reliability weight: 20%
#         )

#         # Filter only relevant rows for low-stock products
#         low_stock_vendors = self.df[self.df['productname'].isin(low_stock_products['productname'])]

#         # Find the best vendor for each product by selecting the one with the lowest RankScore
#         best_vendors = low_stock_vendors.loc[low_stock_vendors.groupby('productname')['RankScore'].idxmin()]

#         # Select relevant columns for the output
#         best_vendors = best_vendors[[
#             'productname', 'New Vendors', 'vendor', 'costprice', 'DeliveryTime', 'ReliabilityScore'
#         ]]

#         return best_vendors





# # Example Usage
# if __name__ == "__main__":
#     # Load the dataset
#     file_path = 'modified_dataset.csv'
#     df = pd.read_csv(file_path)

#     # Create instances of ProductBenchmarking and ProductVendorManager
#     product_benchmarking = ProductBenchmarking(df)
#     product_vendor_manager = ProductVendorManager(df)

#     # Select a product by its ID (e.g., ID 1)
#     product_benchmarking.select_product(product_id=1)

#     # Find similar products based on category and subcategory
#     product_benchmarking.find_similar_products()

#     # Fetch and display benchmark data for the first 3 similar products
#     product_benchmarking.get_benchmark_data(num_products_to_display=3)

#     # Display vendor data and sort by reliability and delivery time
#     product_benchmarking.display_vendors()

#     # Get suggestions for all low-stock products
#     suggestions = product_vendor_manager.suggest_vendors_for_low_stock_products()

#     if not suggestions.empty:
#         print("\nBest Vendors for Low-Stock Products:")
#         print(suggestions)
#     else:
#         print("No low-stock products found.")
        
        
import pandas as pd

class ProductVendorManager:
    def __init__(self, dataframe):
        """Initialize the ProductVendorManager with the given DataFrame."""
        self.df = dataframe

    def suggest_vendors_for_low_stock_products(self):
        """Suggest vendors for all products that are low on stock based on price, delivery time, and reliability score."""
        # Step 1: Identify low-stock products
        low_stock_products = self.df[self.df['stockquantity'] < self.df['reorderthreshold']]

        if low_stock_products.empty:
            print("No products are low on stock.")
            return pd.DataFrame()

        # Step 2: Calculate a ranking score for each vendor based on price, delivery time, and reliability score
        self.df['RankScore'] = (
            self.df['costprice'] * 0.5 +            # Price weight: 50%
            self.df['DeliveryTime'] * 0.3 +        # Delivery time weight: 30%
            (1 - self.df['ReliabilityScore']) * 0.2  # Reliability weight: 20%
        )

        # Step 3: Filter only relevant rows for low stock products
        low_stock_df = self.df[self.df['stockquantity'] < self.df['reorderthreshold']]

        # Step 4: Sort vendors based on the RankScore for low-stock products
        low_stock_df_sorted = low_stock_df.sort_values(by='RankScore')

        return low_stock_df_sorted[['productname', 'New Vendors', 'costprice', 'DeliveryTime', 'ReliabilityScore', 'RankScore']]

# Example usage:
# Assuming `df` is the DataFrame containing your product dataset
df = pd.read_csv("your_product_data.csv")

# Create an instance of ProductVendorManager
manager = ProductVendorManager(df)

# Suggest vendors for low-stock products
suggested_vendors = manager.suggest_vendors_for_low_stock_products()

# Print the suggested vendors
print(suggested_vendors)
