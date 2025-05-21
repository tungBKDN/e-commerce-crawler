import pandas as pd
import utils as u
from activeLearning import EmotionClassifierNB

# Vars
df = u.load_data()
labels = u.get_labels()
ids = u.random_pooling(df, n=40, random_state=40)

annos = u.display(ids, df, labels, ai_mode=True)
print("Updating...")
df = u.update_labeled(df, annos)

model = EmotionClassifierNB()
training_data = u.prepare_data_traning(df)
model.fit(training_data['comments'], training_data['labels'])
df = model.predict_dataframe(df)

# Fix labels
# df = u.fix_labels(df, 0.75)

# test_data, true_labels = u.prepare_test_data(df)
# pred = model.predict(test_data)
# cm = u.get_confusion_matrix(labels, pred, true_labels)
# print("Confusion matrix:")
# print(cm)
# u.plot_confusion_matrix(cm, true_labels)

# Print the number of confident in range [0.4, 1)
satisfied = df[(df['confidence'] >= 0.4) & (df['confidence'] < 1)]
print(f"Number of confident samples: {len(satisfied)}")

# Rows that has confidence > 0.4 and labeled is null, set label same as the predicted and set labeled to True then print the number of samples
print(f"Number of samples that will be labeled: {len(df[(df['confidence'] >= 0.4) & (df['labeled'].isnull())])}")
df.loc[(df['confidence'] >= 0.4) & (df['labeled'].isnull()), 'label'] = df['predicted_label']
df.loc[(df['confidence'] >= 0.4) & (df['labeled'].isnull()), 'labeled'] = True
df.loc[(df['confidence'] >= 0.4), 'confidence'] = 1

# Saving
df.to_csv("./data/LABEL_clean_comments.csv", index=False)
u.plot_confidence(df)

for i in range(0, 8):
    u.get_significant_words(df, 30, i)
