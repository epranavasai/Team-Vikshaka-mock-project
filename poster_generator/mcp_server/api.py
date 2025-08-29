import httpx
from typing import Any, Dict, Optional
from config import USER_AGENT, REQUEST_TIMEOUT, logger

async def make_instagram_request(url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Instagram API request failed: %s", e)
            return None
