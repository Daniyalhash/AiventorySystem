def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return {"error": f"Missing required columns: {', '.join(missing_columns)}"}
    return {"success": True}