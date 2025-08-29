import logging
from typing import Any, Dict

import aiohttp
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_NAME = "mistral"
LLM_API_URL = "http://localhost:11434/api/generate"
MCP_SSE_URL = "http://localhost:8000/sse"


class LLMService:
    """LLM Wrapper client with MCP integration for post-analysis."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, mcp_sse_url: str = MCP_SSE_URL) -> None:
        self.model_name = model_name
        self.mcp_sse_url = mcp_sse_url

    async def _query_llm(self, prompt: str) -> str:
        """Send a prompt to the LLM and return its response."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                LLM_API_URL,
                json={"model": self.model_name, "prompt": prompt, "stream": False},
            ) as resp:
                result: Dict[str, Any] = await resp.json()
                return result.get("response", "")

    async def ainvoke(self, prompt: str) -> str:
        """
        Invoke the LLM.  
        If the LLM response requests tool usage, call the MCP tool,
        then re-run the query with contextualized input.
        """
        first_output = await self._query_llm(prompt)

        if "[TOOL:get_posts" in first_output:
            posts = await self._fetch_posts_with_mcp()

            followup_prompt = (
                f"Product Description: {prompt}\n\n"
                f"Here are the recent Instagram posts and corresponding comments retrieved:\n"
                f"- {posts}\n\n"
                "Each post below contains its caption and the comments it received.\n\n"
                "Analyze the comments to understand engagement "
                "(which tone, style, or phrasing got better responses).\n\n"
                "Based on this analysis, generate a new caption for the given product.\n\n"
                "The new caption should match the tone of past successful posts "
                "and aim to maximize user engagement (likes, comments, shares, saves)."
            )

            return await self._query_llm(followup_prompt)

        return first_output

    async def _fetch_posts_with_mcp(self) -> Any:
        """Fetch posts via the MCP tool using SSE."""
        async with sse_client(self.mcp_sse_url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool("get_posts", {"limit": 3})
                posts = result.content
                logger.debug("Tool output: %s", posts)
                return posts
