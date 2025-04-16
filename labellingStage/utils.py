import pandas as pd

def load_data(file_path: str = "./data/LABEL_clean_comments.csv") -> pd.DataFrame:
    """
    Load the data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found

def random_pooling(df: pd.DataFrame, n: int = 20, random_state = 42) -> list:
    """
    Randomly select n rows from the DataFrame.
    Return: list of indexes
    """
    df = df[df['labeled'] != True].sample(n=n, random_state=random_state)
    return df.index.tolist()

def update_labeled(df: pd.DataFrame, annotations: list, file_name: str = "./data/LABEL_clean_comments.csv") -> pd.DataFrame:
    """
    Update the DataFrame with the annotations.
    """
    for index, annotation in annotations:
        df.at[index, 'label'] = annotation
        df.at[index, 'is_labeled'] = True
        df.at[index, 'confidence'] = 1
    # Save the updated DataFrame to a CSV file
    df.to_csv(file_name, index=False)
    return df

def get_labels(file_name: str = "./data/emotions.txt") -> list:
    """
    Get defined labels
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            labels = [line.strip().split(' ', 1)[1] for line in file.readlines() if ' ' in line]
        return labels
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return []  # Return an empty list if the file is not found

def display(ids: list, df: pd.DataFrame, labels: list):
    """
    Interactive display for input
    """
    total_comments = len(ids)
    annotations = []
    for i in range(total_comments):
        print("\033[H\033[J", end="")  # ANSI escape sequence to clear the console
        print(f"\nComment {i + 1}/{total_comments}:\n")
        print(df.iloc[ids[i]]['comment'])
        print("\nLabels:")
        for j, label in enumerate(labels):
            print(f"{j}. {label}")
        print("\n LABEL: ", end="")
        user_input = input()
        if user_input.isdigit():
            annotations.append((ids[i], user_input))
        else:
            print("Invalid input. Please enter a number corresponding to the label.")
            continue
    return annotations

def load_stopwords(file_name: str = "./dictionary/vietnamese-stopwords.txt") -> set:
    """
    Load stopwords from a file into a set.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            stopwords = {line.strip() for line in file.readlines()}
        return stopwords
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return set()  # Return an empty set if the file is not found

def remove_stopwords(text: str, stopwords: set) -> str:
    """
    Remove stopwords from a given text.
    """
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)