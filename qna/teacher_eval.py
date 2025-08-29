import ollama, json, re
from retriever import retrieve_product

def evaluate_reply_with_teacher(comment, reply, embedder, index, df, threshold=0.7):
    product_info = retrieve_product(comment, embedder, index, df)

    if not product_info:
        return 0.0, "", "Sorry for the inconvenience, but I currently don't have enough context to answer your comment accurately. Could you please specify the product you're referring to?"
    
    prompt = f"""
You are an evaluation expert.
Evaluate the student's reply for correctness, helpfulness, tone, and relevance based on the provided context.

Rules:
Consider ONLY the given product catalog as ground truth.
Also consider whether the product mentioned in the comment are same or not
If product is mismatched then give low score and reject the reply
If the student reply uses correct product info + link + Instagram-friendly tone, give a high score.
Always output ONLY valid JSON. No extra text.

Strictly return valid JSON with the following structure:
{{
  "confidence_score": <number between 0.0 and 1.0>,
  "feedback": "<short, clear feedback>"
}}

Question/Comment: "{comment}"

Product Info:
- ID: {product_info["Product ID"]}
- Product: {product_info["Product Name"]} | Description: {product_info["Product Description"]}
- Brand: {product_info["Product Brand Name"]}
- Sizes: {product_info["Size Availability"]}
- Colours: {product_info["Colors Available"]}
- Stock: {product_info["Stock Availability"]}
- Price: {product_info["Price"]}
- Link: {product_info["Online Store Link"]}

Student Reply: "{reply}"
"""

    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
    raw_output = response['message']['content'].strip()

    json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
    if json_match:
        try:
            parsed_output = json.loads(json_match.group(0))
            score = float(parsed_output.get("confidence_score", 0.0))
            feedback = parsed_output.get("feedback", "").strip()
        except json.JSONDecodeError:
            score, feedback = 0.0, "Invalid JSON format."
    else:
        score, feedback = 0.0, "No valid JSON found."

    print(f"Teacher evaluation score = {score}")
    print(f"Debug: Teacher evaluation output = {raw_output}")
    final_reply = reply if score >= threshold else "Sorry for the inconvenience, but I currently don't have enough context to answer your comment accurately."
    return score, feedback, final_reply
