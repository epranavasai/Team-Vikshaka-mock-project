import os
from typing import Any, Dict, List, cast
from api import make_instagram_request
from models import Comment
from config import INSTAGRAM_API_BASE, logger

async def get_post_engagement(post_id: str) -> Dict[str, Any]:
    """
    Fetch engagement insights for a given post and compute engagement score.
    
    Args:
        post_id: Instagram post ID.
    
    Returns:
        Engagement metrics and score.
    """
    data = await make_instagram_request(
        f"{INSTAGRAM_API_BASE}/{post_id}/insights",
        {
            "metric": "reach,shares,saved,likes,comments",
            "access_token": os.getenv("ACCESS_TOKEN"),
        },
    )

    logger.debug("Engagement data for post %s: %s", post_id, data)

    if not data or "data" not in data:
        return {"observation": "Failed to fetch engagement data."}

    metrics = {item["name"]: item["values"][0]["value"] for item in data["data"]}
    reach = metrics.get("reach", 0)
    likes = metrics.get("likes", 0)
    comments = metrics.get("comments", 0)
    shares = metrics.get("shares", 0)
    saved = metrics.get("saved", 0)

    engagement_score = (likes + comments + shares + saved) / max(reach, 1)

    return {
        "metrics": metrics,
        "engagement_score": round(engagement_score, 4),
    }


async def get_comments_for_post(post_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    Fetch comments for a given Instagram post.
    
    Args:
        post_id: Instagram post ID.
        limit: Max number of comments.
    
    Returns:
        Dict with comments or failure observation.
    """
    data = await make_instagram_request(
        f"{INSTAGRAM_API_BASE}/{post_id}/comments",
        {
            "access_token": os.getenv("ACCESS_TOKEN"),
            "fields": "id,text,username,timestamp",
            "limit": limit,
        },
    )

    if not data or "data" not in data:
        return {"observation": "Failed to fetch comments."}

    return {"comments": data["data"]}


async def format_posts(data: Dict[str, Any]) -> List[str]:
    """
    Format fetched posts into a readable list with comments.
    
    Args:
        data: Raw Instagram API response for posts.
    
    Returns:
        List of formatted post strings.
    """
    formatted = []
    for i, post in enumerate(data.get("data", []), start=1):
        caption = post.get("caption", "").strip()
        timestamp = post.get("timestamp")
        post_id = post.get("id")

        # Fetch comments
        comments_data: Dict[str, Any] = await get_comments_for_post(post_id, limit=3)
        comments: List[Comment] = cast(List[Comment], comments_data.get("comments", []))

        comments_str = "\n".join(
            f"    - {c['text']}" for c in comments
        ) if comments else "    No comments found."

        formatted.append(
            f"Post {i} ({timestamp}): {caption}\n  Comments:\n{comments_str}"
        )

    return formatted

