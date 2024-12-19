from django.urls import path
from .views import signup, login, validate_token,upload_dataset,complete_signup,get_user_details,get_total_products,product_benchmark

urlpatterns = [
    path('signup/', signup, name='signup'),
    path('login/', login, name='login'),
    path('validate-token/', validate_token, name='validate_token'),
    path('upload_dataset/', upload_dataset, name='upload_dataset'),  # Add this line
    path('complete_signup/', complete_signup, name='complete_signup'),  # Add this line
    path('get-user-details/', get_user_details, name='get_user_details'),
    path('get-total-products/',get_total_products, name='get_total_products'),
    path('product-benchmark/',product_benchmark, name='product_benchmark'),

]
