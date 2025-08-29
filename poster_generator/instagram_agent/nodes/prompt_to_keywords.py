from ..state import State
from ..config import logger

async def prompt_to_keywords(state: State) -> State:
    product_desc = state["product_description"]
    state["keywords"] = product_desc.split()
    logger.debug("Keywords after prompt split: %s", state["keywords"])
    return state
