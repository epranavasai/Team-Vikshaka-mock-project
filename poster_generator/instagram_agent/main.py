import asyncio
from .graph_builder import build_graph
from .state import State
from .config import logger
from .services.poster_image_service import poster_generation
from post_scheduler.main import main as post_scheduler

async def run_chat():
    graph = build_graph()
    logger.info("MCP LLM Chat — type 'exit' to quit.")

    while True:
        desc = input("Enter product description: ").strip()
        if desc.lower() in ("exit", "quit"):
            break

        image_url = input("Enter image URL: ").strip()
        brand = input("Enter brand name: ").strip()

        state: State = State(
            product_description=desc,
            keywords=[],
            generated_caption="",
            final_output="",
            image_url=image_url,
            brand=brand,
        )

        final = await graph.ainvoke(state)

        print("\nGenerated Caption:")
        print(final["final_output"])

        try:
            poster_generation(desc, brand, image_url)
        except Exception as e:
            logger.error("Poster generation failed: %s", e)

        try:
            post_scheduler(False, None, state["product_description"])
        except Exception as e:
            logger.error("Post scheduling failed: %s", e)

        break