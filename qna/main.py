from data_loader import load_data
from embedder import build_index
from pipeline import generate_final_reply

if __name__ == "__main__":
    csv_path = "vikshaka_products_with_description.csv"
    df = load_data(csv_path)
    embedder, index = build_index(df)

while True:
    # Get user input
    comment = input("Enter a comment: ")

    # Exit condition
    if comment.lower() in ["exit", "quit", "bye"]:
        print("Chat ended. Goodbye!")
        break

    # Generate response
    final_reply = generate_final_reply(comment, embedder, index, df) # type: ignore

    # Display reply
    print("Bot Reply:", final_reply)

