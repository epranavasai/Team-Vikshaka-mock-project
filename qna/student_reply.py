import ollama
from retriever import retrieve_product

def generate_reply(comment, embedder, index, df):
    product_info = retrieve_product(comment, embedder, index, df)

    if not product_info:
        return "Sorry, I couldn’t find this product in our catalog."

    prompt = f"""
You are a helpful, friendly, and professional social media assistant for a fashion brand.  
Your job is to write short, stylish, and polite replies to user comments or questions on social media posts.  

Instructions:
- Always read the **customer's comment first** and respond directly to it.  
- Keep it concise — **1 to 2 sentences only**.  
- Use a warm, approachable, and trendy tone.  
- Mention product details (name, brand, sizes, colors, price) **only if they are relevant to the exact comment**.  
- If the customer asks about a color/size/availability, check the product info and answer clearly.  
- If the requested option is not available, politely suggest an alternative from the given product details.  
- Provide the clickable product link **only if it is helpful for the customer’s query**.  
- Never include placeholders like [Link] or [URL]; always use the actual link.  
- Do not talk about any product that is not mentioned in the **Product Info** section.  
- In answer give link at the end of the reply once.
Customer Comment: "{comment}"

Product Info:
- ID: {product_info.get('Product ID', '')}
- Product: {product_info.get('Product Name', '')} | Product Description: {product_info.get('Product Description', '')}
- Brand: {product_info.get('Product Brand Name', '')}
- Sizes: {product_info.get('Size Availability', '')}
- Colours: {product_info.get('Colors Available', '')}
- Stock: {product_info.get('Stock Availability', '')}
- Price: {product_info.get('Price', '')}
- Link: {product_info.get('Online Store Link', '')}
"""

    response = ollama.chat(model="mistral", messages=[
        {"role": "system", "content": "You are a helpful social media assistant."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content'].strip()
