from langgraph.graph import StateGraph, START, END
from .state import State
from .nodes import prompt_to_keywords, image_to_keywords, generate_instagram_content, combine

def build_graph():
    builder = StateGraph(State)
    builder.add_node(prompt_to_keywords.prompt_to_keywords)
    builder.add_node(image_to_keywords.image_to_keywords)
    builder.add_node(generate_instagram_content.generate_instagram_content)
    builder.add_node(combine.combine)

    builder.add_edge(START, prompt_to_keywords.prompt_to_keywords.__name__)
    builder.add_edge(prompt_to_keywords.prompt_to_keywords.__name__, image_to_keywords.image_to_keywords.__name__)
    builder.add_edge(image_to_keywords.image_to_keywords.__name__, generate_instagram_content.generate_instagram_content.__name__)
    builder.add_edge(generate_instagram_content.generate_instagram_content.__name__, combine.combine.__name__)
    builder.add_edge(combine.combine.__name__, END)

    return builder.compile()
