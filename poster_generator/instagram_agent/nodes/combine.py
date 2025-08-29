from ..state import State

async def combine(state: State) -> State:
    state["final_output"] = state["generated_caption"]
    return state
