from ..state import State
from ..config import logger
from ..services.llm_service import LLMService

async def generate_instagram_content(state: State) -> State:
    llm = LLMService("mistral")
    prod = state["product_description"]
    keys = state["keywords"]

    prompt = f"""
        You are an Instagram caption generator with access to an MCP tool named `get_posts`.

        Your process is ALWAYS two steps:

        Step 1:
            Output exactly:
            [TOOL:get_posts limit=3]
            Do not add any extra words, explanation, or caption in this step.

        Step 2:
            Once you have the tool results, use them to generate a catchy Instagram caption
            in the same tone and style as the recent posts.

        ---
        Product Description: {prod}
        Keywords: {keys}

        Important:
        - If you have not yet called the tool, do not generate the caption.
        - Only after seeing the tool results should you create the caption.
    """

    try:
        response = await llm.ainvoke(prompt)
        state["generated_caption"] = response
        logger.debug("Generated caption: %s", response)
    except Exception as e:
        logger.error("Failed to generate Instagram content: %s", e)
        state["generated_caption"] = ""
    return state
