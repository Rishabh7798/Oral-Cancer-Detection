import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

dir = 'test&train'
categories = ['cancer', 'non-cancer']
image_size = (100, 100)

data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        oc_img = cv2.imread(imgpath, 0)

        try:
            oc_img = cv2.resize(oc_img, image_size)
            image = np.array(oc_img).flatten()
            data.append([image, label])
        except Exception as e:
            pass

random.shuffle(data)

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

Xtrain, Xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.3)

model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(Xtrain, ytrain)

accuracy = model.score(Xtest, ytest)
print('Accuracy:', accuracy)

# Save the model
with open('model.sav', 'wb') as file:
    pickle.dump(model, file)

# Perform prediction and visualization
prediction = model.predict(Xtest)
categories = ['cancer', 'non-cancer']
print('Prediction:', categories[prediction[0]])

oralCancer = Xtest[0].reshape(*image_size)
plt.imshow(oralCancer, cmap='gray')
plt.show()

accuracy = accuracy_score(ytest, prediction)
precision = precision_score(ytest, prediction)
f1 = f1_score(ytest, prediction)
recall = recall_score(ytest, prediction)

print('Accuracy Score:', accuracy)
print('Precision Score:', precision)
print('F1 Score:', f1)
print('Recall Score:', recall)

# Plotting linear graph
metrics = ['Accuracy', 'Precision', 'F1', 'Recall']
scores = [accuracy, precision, f1, recall]

# Plotting graph for accuracy and F1 score
metrics = ['Accuracy', 'F1']
scores = [accuracy, f1]

plt.plot(metrics, scores)
plt.xlabel('Metrics')
plt.ylabel('Scores')
plt.title('Accuracy and F1 Score')
plt.show()