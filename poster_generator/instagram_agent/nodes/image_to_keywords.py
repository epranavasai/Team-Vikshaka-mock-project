from ..state import State
from ..config import DEVICE, FASHION_MODEL_NAME, logger
from ..services.image_service import extract_keywords_from_image

async def image_to_keywords(state: State) -> State:
    try:
        extra_keywords = await extract_keywords_from_image(state["image_url"])
        state["keywords"].extend(extra_keywords)
    except Exception as e:
        logger.error("Image-to-keywords failed: %s", e)
    return state
