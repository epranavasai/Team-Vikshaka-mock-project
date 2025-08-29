from typing import TypedDict, List

class State(TypedDict):
    product_description: str
    keywords: List[str]
    generated_caption: str
    final_output: str
    image_url: str
    brand: str
