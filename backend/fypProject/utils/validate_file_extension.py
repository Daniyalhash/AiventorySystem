def validate_file_extension(file_name):
    allowed_extensions = ['csv']
    file_extension = file_name.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        raise ValidationError(f"Invalid file type. Only {', '.join(allowed_extensions)} files are allowed.")
