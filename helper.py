import os

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
        return []  # Return an empty list if the file does not exist

    print(f"Reading hrefs from {FILE_NAMES['hrefs']}...")
    with open(FILE_NAMES["hrefs"], "r") as file:
        hrefs = file.readlines()
        return [href.strip() for href in hrefs if href.strip()]