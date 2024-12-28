from django.urls import path
from .views import signup, login, validate_token,upload_dataset,complete_signup,get_user_details,get_total_products,product_benchmark,get_current_dataset,get_inventory_visuals,get_insights_visuals,get_vendor,get_categories,get_top_products_by_category,get_products_by_name,get_low_stock_products,get_categories_p,get_lowVendor,get_vendor_details

urlpatterns = [
    path('signup/', signup, name='signup'),
    path('login/', login, name='login'),
    path('validate-token/', validate_token, name='validate_token'),
    path('upload_dataset/', upload_dataset, name='upload_dataset'),  # Add this line
    path('complete_signup/', complete_signup, name='complete_signup'),  # Add this line
    path('get-user-details/', get_user_details, name='get_user_details'),
    path('get-total-products/',get_total_products, name='get_total_products'),
    path('product-benchmark/',product_benchmark, name='product_benchmark'),
    path('get-current-dataset/',get_current_dataset, name='get_current_dataset'),
    path('get-inventory-visuals/',get_inventory_visuals, name='get_inventory_visuals'),
    path('get-insights-visuals/',get_insights_visuals, name='get_insights_visuals'),
    path('get-vendor/',get_vendor, name='get_vendor'),
    
    path('get-top-products-by-category/',get_top_products_by_category, name='get_top_products_by_category'),
    path('get-categories/',get_categories, name='get_categories'),
        path('get-categories-p/',get_categories_p, name='get_categories_p'),
    path('get-lowVendor/',get_lowVendor, name='get_lowVendor'),

        path('get-products-by-name/',get_products_by_name, name='get_products_by_name'),
        # path('get-vendor-by-id/',get_vendor_by_id, name='get_vendor_by_id'),
    path('get-low-stock-products/',get_low_stock_products, name='get_low_stock_products'),
        path('get-vendor-details/',get_vendor_details, name='get_vendor_details')


]
