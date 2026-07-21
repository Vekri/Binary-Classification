#!/usr/bin/env python3
"""Create or trigger Render deploy via API. Requires RENDER_API_KEY env var."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

API = "https://api.render.com/v1"
REPO = "https://github.com/Vekri/Binary-Classification"
BRANCH = "master"
SERVICE_NAME = "binary-classification-ml"


def request(method: str, path: str, body: dict | None = None) -> dict:
    key = os.environ.get("RENDER_API_KEY")
    if not key:
        print("Set RENDER_API_KEY (Render Dashboard → Account Settings → API Keys)")
        sys.exit(1)

    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print(exc.read().decode())
        raise


def list_services() -> list[dict]:
    payload = request("GET", "/services?limit=50")
    return payload if isinstance(payload, list) else payload.get("services", [])


def find_service(services: list[dict]) -> dict | None:
    for item in services:
        svc = item.get("service") or item
        if svc.get("name") == SERVICE_NAME:
            return svc
        repo = (svc.get("repo") or "").lower()
        if "binary-classification" in repo:
            return svc
    return None


def create_service(owner_id: str) -> dict:
    body = {
        "type": "web_service",
        "name": SERVICE_NAME,
        "ownerId": owner_id,
        "repo": REPO,
        "branch": BRANCH,
        "autoDeploy": "yes",
        "serviceDetails": {
            "env": "python",
            "envSpecificDetails": {
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": (
                    "streamlit run app/main.py "
                    "--server.port=$PORT --server.address=0.0.0.0 "
                    "--server.headless=true --browser.gatherUsageStats=false"
                ),
            },
            "healthCheckPath": "/_stcore/health",
            "plan": "free",
            "region": "oregon",
        },
    }
    return request("POST", "/services", body)


def trigger_deploy(service_id: str) -> dict:
    return request("POST", f"/services/{service_id}/deploys", {"clearCache": "do_not_clear"})


def main() -> None:
    owner_id = os.environ.get("RENDER_OWNER_ID")
    services = list_services()
    svc = find_service(services)

    if svc:
        print(f"Found service: {svc['name']} ({svc['id']})")
        deploy = trigger_deploy(svc["id"])
        print("Deploy triggered:", json.dumps(deploy, indent=2))
        return

    if not owner_id:
        print("Service not found. Set RENDER_OWNER_ID (workspace ID from Render settings).")
        print("Or use one-click deploy:", f"https://render.com/deploy?repo={REPO}/tree/{BRANCH}")
        sys.exit(1)

    created = create_service(owner_id)
    print("Service created:", json.dumps(created, indent=2))


if __name__ == "__main__":
    main()
