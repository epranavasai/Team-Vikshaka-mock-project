import os
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP
from services import format_posts, get_comments_for_post
from api import make_instagram_request
from config import INSTAGRAM_API_BASE, logger

mcp = FastMCP(name="instagram", host="0.0.0.0", port=8000)

@mcp.tool()
async def get_posts(limit: int = 5) -> Dict[str, Any]:
    url = f"{INSTAGRAM_API_BASE}/{os.getenv('IG_USER_ID')}/media"
    params = {
        "access_token": os.getenv("ACCESS_TOKEN"),
        "fields": "id,caption,media_url,permalink,timestamp",
        "limit": limit,
    }

    data = await make_instagram_request(url, params)
    if data is None:
        logger.warning("Failed to fetch posts from Instagram API")
        return {"observation": ["No posts found or error occurred."]}

    formatted_posts = await format_posts(data)
    return {"observation": formatted_posts}
