import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

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
        df.at[index, 'labeled'] = True
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

def prepare_data_traning(df: pd.DataFrame) -> dict:
    """
    Prepare data for training
    """
    # Get comment_nonsw if labeled is True
    comments = df[df['labeled'] == True]['comment_nonsw'].tolist()
    labels = df[df['labeled'] == True]['label'].tolist()
    return {
        "comments": comments,
        "labels": labels
    }

def prepare_data_predict(df: pd.DataFrame):
    """
    Prepare data for prediction
    """
    # Get comment_nonsw from idx in df that is not labeled
    idx = df[df['labeled'] != True].index.tolist()
    comments = df.loc[idx, 'comment_nonsw'].tolist()
    return {
        "idx": idx,
        "comments": comments
    }

def plot_confidence(df: pd.DataFrame):
    """
    Plot confidence scores
    """
    # Filter out rows where 'confidence' is NaN or not a number
    df = df[df['confidence'].notna() & (df['confidence'] != 0)]
    total_labeled = len(df[df['labeled'] == True])

    mean_confidence = df[df['labeled'] != True]['confidence'].mean()
    med_confidence = df[df['labeled'] != True]['confidence'].median()
    print(f"Mean confidence: {mean_confidence}")
    print(f"Median confidence: {med_confidence}")

    # Append to file: ./data/history.csv
    text_to_append = f"{total_labeled},{mean_confidence},{med_confidence}\n"
    with open("./data/history.csv", "a") as file:
        file.write(text_to_append)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(df['confidence'], bins=50, alpha=0.7, color='blue')
    plt.title(f'Confidence Scores Distribution - Labeled: {str(total_labeled).zfill(4)} | {mean_confidence}, {med_confidence}')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.ylim(0, 5000)  # Set y-axis limits
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f"./images/confidence-{str(total_labeled).zfill(4)}.png", dpi=300, bbox_inches='tight')
    plt.show()

def get_significant_words(df: pd.DataFrame, n: int = 30, label = 0) -> dict:
    """
    Get significant words from the DataFrame
    """
    # Filter comments by label and labeled = True
    filtered_comments = df[(df['labeled'] == True) & (df['label'] == label)]['comment_nonsw']

    # Tokenize and count word frequencies
    word_counter = Counter()
    for comment in filtered_comments:
        word_counter.update(comment.split())

    # Get the top n significant words
    significant_words = dict(word_counter.most_common(n))

    # Plot a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(significant_words)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Top {n} Significant Words for Label {label}")
    plt.savefig(f"./images/significant_words/significant_words_{label}.png", dpi=300, bbox_inches='tight')

    return significant_words