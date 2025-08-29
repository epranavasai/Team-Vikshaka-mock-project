import os
import logging
from dotenv import load_dotenv

# Load env
load_dotenv()

INSTAGRAM_API_BASE = "https://graph.facebook.com/v23.0"
USER_AGENT = "mistral-7b-caption-generator/1.0"
REQUEST_TIMEOUT = 30.0

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_tool")
