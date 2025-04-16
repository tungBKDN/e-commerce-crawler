import pandas as pd
import utils as u

# Vars
df = u.load_data()
# labels = u.get_labels()
# ids = u.random_pooling(df, n=3)

# annos = u.display(ids, df, labels)
# print("Updating...")
# df = u.update_labeled(df, annos)

stopwords = u.load_stopwords()
df['comment_nonsw'] = df['comment'].apply(lambda x: u.remove_stopwords(x, stopwords))
print("Stopwords removed")
print(df.head(10))
# Save
df.to_csv("./data/LABEL_clean_comments.csv", index=False)