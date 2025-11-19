import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from config import config

FaceFrame = pd.read_csv('csv_dataset/kaggle_dataset.csv')
X, y = FaceFrame['image_name'], FaceFrame['age']
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=config['seed'], stratify=y)

train_dataset = pd.concat([X_train, y_train], axis = 1)
test_dataset = pd.concat([X_test, y_test], axis = 1)
train_dataset,valid_dataset = train_test_split(train_dataset,train_size=0.8, random_state=config['seed'],stratify=y_train)

# display stratified datasets for training and test
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].hist(train_dataset.age, bins=len(train_dataset.age.unique())); axes[0].set_title('Train'); axes[0].set_xlabel('Age'); axes[0].set_ylabel('Frequency')
axes[1].hist(test_dataset.age, bins=len(test_dataset.age.unique())); axes[1].set_title('Test'); axes[1].set_xlabel('Age'); axes[1].set_ylabel('Frequency')
plt.show()
axes[2].hist(valid_dataset.age, bins=len(valid_dataset.age.unique())); axes[2].set_title('Valid'); axes[2].set_xlabel('Age'); axes[2].set_ylabel('Frequency')
plt.show()

# Uncomment when ready to save
# train_dataset.to_csv('./csv_dataset/train_set.csv', index=False)
# test_dataset.to_csv('./csv_dataset/test_set.csv', index=False)
# valid_dataset.to_csv('./csv_dataset/valid_set.csv', index=False)