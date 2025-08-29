from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os

load_dotenv()

# Create an MCP server
mcp = FastMCP(
    name="instagram",
    host="0.0.0.0",  # only used for SSE transport (localhost)
    port=8000,  # only used for SSE transport (set this to any port)
)

# Constants
instagram_API_BASE = f"https://graph.instagram.com/v23.0/{os.environ.get('IG_USER_ID')}"
USER_AGENT = "mistral-7b-caption-generator/1.0"


async def make_instagram_request(url: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """Make a request to the instagram API with proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_posts(data: dict):
    """Format fetch posts response into a readable string."""
    formatted = []
    for i, post in enumerate(data["data"], start=1):
        caption = post.get("caption", "").strip()
        formatted.append(f"Post {i} ({post.get('timestamp')}): {caption}")
    return {"posts_text": "\n".join(formatted)}

@mcp.tool()
async def get_posts(limit: int = 5):
    """Get posts from instagram.

    Args:
        limit: numner of posts to fetch
    """
    url = f"{instagram_API_BASE}/media"
    params = {
        "fields": "id,caption,media_url,permalink,timestamp",
        "access_token": os.environ.get('ACCESS_TOKEN'),  # Ensure you have set this environment variable
        "limit": limit
    }
    data = await make_instagram_request(url, params)
    return format_posts(data) if data else {"error": "Failed to fetch posts"}

# Run the server
if __name__ == "__main__":
    transport = "sse"
    if transport == "stdio":
        print("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        print("Running server with SSE transport")
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unknown transport: {transport}")