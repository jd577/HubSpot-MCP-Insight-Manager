import sys
import io
import os
import httpx
from typing import Any
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()

mcp = FastMCP("HubSpot CRM Server")


HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUBSPOT_BASE_URL = "https://api.hubapi.com/crm/v3"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

async def make_request(
    url: str,
    params: dict = None,
    headers: dict = None,
    method: str = "GET",
    json_data: dict = None
) -> dict[str, Any] | None:
    """Standardized async API request wrapper with error handling."""
    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, params=params, headers=headers, timeout=15.0)
            elif method.upper() == "POST":
                response = await client.post(url, json=json_data, headers=headers, timeout=15.0)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"⚠️ API Connection Error: {e}", file=sys.stderr)
            return None

@mcp.tool()
async def get_contacts(limit: int = 10) -> str:
    """Fetch recent contacts from HubSpot CRM."""
    headers = {"Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}"}
    params = {"limit": limit, "properties": ["firstname", "lastname", "email"]}
    
    url = f"{HUBSPOT_BASE_URL}/objects/contacts"
    data = await make_request(url, params=params, headers=headers)

    if not data or "results" not in data:
        return "Could not retrieve contacts."

    contacts = [f"- {c['properties'].get('firstname', '')} {c['properties'].get('lastname', '')} ({c['properties'].get('email', 'N/A')})" 
                for c in data["results"]]
    return f"Found {len(contacts)} contacts:\n" + "\n".join(contacts)

@mcp.tool()
async def analyze_crm_data(query: str) -> str:
    """Analyze HubSpot CRM metrics using Groq AI for instant business insights."""
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY missing from environment."

    headers = {"Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}"}
    contacts = await make_request(f"{HUBSPOT_BASE_URL}/objects/contacts", params={"limit": 10}, headers=headers)
    deals = await make_request(f"{HUBSPOT_BASE_URL}/objects/deals", params={"limit": 10}, headers=headers)

    crm_context = {
        "contacts": contacts.get("results", []) if contacts else [],
        "deals": deals.get("results", []) if deals else []
    }

    groq_payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a professional CRM analyst. Provide concise ASCII-only insights."},
            {"role": "user", "content": f"Query: {query}\n\nData: {crm_context}"}
        ]
    }

    groq_response = await make_request(f"{GROQ_BASE_URL}/chat/completions", 
                                       method="POST", 
                                       json_data=groq_payload, 
                                       headers={"Authorization": f"Bearer {GROQ_API_KEY}"})

    if groq_response and "choices" in groq_response:
        analysis = groq_response["choices"][0]["message"]["content"]
        return f"AI Analysis:\n{analysis.encode('ascii', 'ignore').decode('ascii')}"
    
    return "Failed to generate AI analysis."

def main():
    """Server entry point using Standard Input/Output (stdio) transport."""
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()