import os
import boto3
from dotenv import load_dotenv

load_dotenv()
bedrock = boto3.client(
    "bedrock-runtime",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-west-2"),
)

resp = bedrock.list_models()
print("Available models in your account:")
for m in resp.get("modelSummaries", []):
    # Only show embedding-capable ones
    if m.get("primaryContainer", {}).get("modelType") == "EMBEDDING":
        print(" •", m["modelId"])