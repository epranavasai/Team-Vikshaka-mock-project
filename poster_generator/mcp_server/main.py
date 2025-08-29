from tools import mcp
from config import logger

if __name__ == "__main__":
    transport = "sse"
    if transport == "stdio":
        logger.info("Running server with stdio transport")
        mcp.run(transport="stdio")
    elif transport == "sse":
        logger.info("Running server with SSE transport")
        mcp.run(transport="sse")
    else:
        raise ValueError(f"Unknown transport: {transport}")
