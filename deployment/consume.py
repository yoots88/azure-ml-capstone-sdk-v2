import argparse
import json
import sys
from pathlib import Path

import requests
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validate import validate_payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--endpoint", required=True)
    args = parser.parse_args(argv)

    client = MLClient(DefaultAzureCredential(), args.subscription,
                      args.resource_group, args.workspace)
    endpoint = client.online_endpoints.get(args.endpoint)
    key = client.online_endpoints.get_keys(args.endpoint).primary_key
    request_file = Path(__file__).with_name("sample-request.json")
    payload = json.loads(request_file.read_text(encoding="utf-8"))
    schema_path = request_file.with_name("request-schema.json")
    numeric_columns = None
    if schema_path.is_file():
        numeric_columns = json.loads(schema_path.read_text(encoding="utf-8"))["numeric_columns"]
    validate_payload(payload, numeric_columns=numeric_columns)
    response = requests.post(
        endpoint.scoring_uri, json=payload,
        headers={"Authorization": f"Bearer {key}"}, timeout=60,
    )
    print(f"HTTP {response.status_code}")
    response.raise_for_status()
    body = response.json()
    labels = body.get("predictions") if isinstance(body, dict) else body
    if not isinstance(labels, list) or len(labels) != len(payload["input_data"]["data"]):
        raise RuntimeError(f"Unexpected prediction shape: {body}")
    if any(label not in (0, 1, "0", "1") for label in labels):
        raise RuntimeError(f"Unexpected labels: {labels}")
    print(json.dumps(body, indent=2))

if __name__ == "__main__":
    main()
