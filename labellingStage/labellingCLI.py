import pandas as pd
import utils as u
from activeLearning import EmotionClassifierNB

# Vars
df = u.load_data()
labels = u.get_labels()
ids = u.random_pooling(df, n=20)

annos = u.display(ids, df, labels)
print("Updating...")
df = u.update_labeled(df, annos)

model = EmotionClassifierNB()
training_data = u.prepare_data_traning(df)
model.fit(training_data['comments'], training_data['labels'])
df = model.predict_dataframe(df)

# Saving
df.to_csv("./data/LABEL_clean_comments.csv", index=False)
u.plot_confidence(df)

for i in range(0, 8):
    u.get_significant_words(df, 30, i)