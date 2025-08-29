import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def suggest_posting_windows(model, description, top_n=1, window_days=1, season_span=30):
    all_days = pd.DataFrame({"day_of_year": np.arange(1, 366)})
    all_days["sin_day"] = np.sin(2 * np.pi * all_days["day_of_year"] / 365)
    all_days["cos_day"] = np.cos(2 * np.pi * all_days["day_of_year"] / 365)
    all_days["tot_desc"] = description

    preds = model.predict(all_days[["tot_desc", "sin_day", "cos_day"]])
    all_days["pred_engagement"] = preds
    ranked = all_days.sort_values("pred_engagement", ascending=False)

    windows, used = [], []
    for _, row in ranked.iterrows():
        doy = int(row["day_of_year"])
        score = row["pred_engagement"]
        if all(abs(doy - u) >= season_span for u in used):
            start = ((doy - window_days - 1) % 365) + 1
            end = ((doy + window_days - 1) % 365) + 1
            start_dt = datetime(2021, 1, 1) + timedelta(days=start-1)
            end_dt = datetime(2021, 1, 1) + timedelta(days=end-1)
            windows.append({
                "window": f"{start_dt:%b %d} – {end_dt:%b %d}",
                "predicted_reviews": float(score)
            })
            used.append(doy)
        if len(windows) >= top_n:
            break
    return windows
