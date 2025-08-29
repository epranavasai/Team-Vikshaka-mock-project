import pandas as pd # type: ignore

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['description'] = df.apply(lambda row: 
        f"Product ID: {row['Product ID']}. "
        f"Name: {row['Product Name']} described as '{row['Product Description']}' from brand {row['Product Brand Name']}. "
        f"Available sizes: {row['Size Availability']}. "
        f"Available colours: {row['Colors Available']}. "
        f"Stock details: {row['Stock Availability']}. "
        f"Price: {row['Price']}. "
        f"Store link: {row['Online Store Link']}.", 
        axis=1
    )
    return df
