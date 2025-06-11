import os
import logging
import json

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_foundation_models(bedrock_client):
    """
    Gets a list of available Amazon Bedrock foundation models.

    :return: The list of available bedrock foundation models.
    """

    try:
        response = bedrock_client.list_foundation_models()
        models = response["modelSummaries"]
        logger.info("Got %s foundation models.", len(models))
        return models

    except ClientError:
        logger.error("Couldn't list foundation models.")
        raise


def main():
    """Entry point for the example. Change aws_region to the AWS Region
    that you want to use."""

    aws_region = "us-east-1"

    bedrock_client = boto3.client(
        service_name="bedrock",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=aws_region,
    )

    fm_models = list_foundation_models(bedrock_client)
    for model in fm_models:
        if "Llama" in model["modelName"]:
            print(f"Model: {model["modelName"]}")
            print(json.dumps(model, indent=2))
            print("---------------------------\n")

    logger.info("Done.")


if __name__ == "__main__":
    main()
