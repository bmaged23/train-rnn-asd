"""Synchronize only this project's public Docker Hub metadata."""
import json
import os
import urllib.request
from pathlib import Path


def request(path, data=None, token=None, method=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request("https://hub.docker.com" + path,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def main():
    token = request("/v2/auth/token", {"identifier": "bmaged23", "secret": os.environ["DOCKERHUB_TOKEN"]})["access_token"]
    path = "/v2/repositories/bmaged23/train-rnn-asd/"
    metadata = request(path, token=token)
    if metadata.get("is_private") is not False:
        raise RuntimeError("Expected public toolkit repository; refusing to change visibility implicitly")
    overview = Path("docs/deployment/DOCKERHUB.md").read_text()
    request(path, {"description": "Checkpoint-required visual ASD API/CLI. CPU, pinned runtime, tested container delivery.",
                   "full_description": overview}, token=token, method="PATCH")
    if request(path).get("full_description") != overview:
        raise RuntimeError("Anonymous Hub overview verification failed")
    print("Public Docker Hub overview verified with GitHub/API/Docker guide links.")


if __name__ == "__main__":
    main()
