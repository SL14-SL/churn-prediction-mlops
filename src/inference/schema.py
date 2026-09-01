import json

def load_feature_schema(path="models/feature_schema.json"):
    """Load the serialized model feature schema from disk."""
    with open(path, "r") as f:
        return json.load(f)
    
    