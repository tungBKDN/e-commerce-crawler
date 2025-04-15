import os
import pandas as pd

FILE_NAMES = {
    "hrefs": "./data/hrefs.txt",
    "productState": "./data/productState.txt",
}

def get_current_product() -> int:
    """
    Get the first product that has not been crawled yet.
    """
    if not os.path.exists(FILE_NAMES["productState"]):
        return 0  # Default to 0 if the file does not exist

    with open(FILE_NAMES["productState"], "r") as file:
        data = file.read().strip()
        return int(data) if data.isdigit() else 0

def update_current_product(product_id: int) -> None:
    """
    Update the current product to the given product_id.
    """
    with open(FILE_NAMES["productState"], "w") as file:
        file.write(str(product_id))

def get_hrefs() -> list:
    """
    Get the hrefs of all products in the database.
    """
    if not os.path.exists(FILE_NAMES["hrefs"]):
        print(f"File {FILE_NAMES['hrefs']} does not exist.")
        return None  # Return an empty list if the file does not exist

    df = pd.read_csv(FILE_NAMES["hrefs"])
    print(f"Loaded {len(df)} hrefs from {FILE_NAMES['hrefs']}.")
    return df["link"].tolist() if "link" in df.columns else None