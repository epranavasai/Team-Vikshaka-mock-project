import pandas as pd
from .model import load_model
from .suggest import suggest_posting_windows
 
print("🚀 main.py started running...")

def main(train=False, data_path=None, desc=None):

    # ===== Training Mode =====
    #if args.train:
    #    if not args.data:
    #        raise ValueError("⚠️ You must provide --data <csv_path> when training.")
    #    print("🔄 Training started...")
    #    reviews_expanded = pd.read_csv(data_path)
    #    X_train, X_test, y_train, y_test = prepare_data(reviews_expanded)
    #    train_and_save_model(X_train, y_train)
    #    print("✅ Training completed and model saved!")

    # ===== Prediction Mode =====
    if desc:
        print(f"🔮 Predicting posting windows for: \"{desc}\"")
        model = load_model()
        top_windows = suggest_posting_windows(model, desc, top_n=1)
        print("📅 Suggested posting windows:")
        for w in top_windows:
            print(f"- {w['window']}")

    # ===== No args provided =====
    if not train and not desc:
        print("⚠️ No action taken. Use --train to train or --desc to predict posting windows.")

if __name__ == "__main__":
    main()
