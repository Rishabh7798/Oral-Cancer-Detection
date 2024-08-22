import tkinter as tk
from tkinter import filedialog
from skimage.feature import hog
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

# Function to extract HOG features from an image
def extract_hog_features(img):
    resized_img = cv2.resize(img, (64, 128))
    resized_img = cv2.convertScaleAbs(resized_img)  # Convert to 8-bit unsigned integer
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    fd = hog(gray_img, feature_vector=True)
    return fd

# Function to preprocess the image data
def preprocess_image(img):
    normalized_img = img.astype(float) / 255.0  # Normalize image
    return normalized_img

# Function to handle button click event
def process_images():
    # Load and preprocess the training images for KNN
    train_data_knn = []
    train_labels_knn = []

    cancer_train_path = 'D:\\oral cancer project\\train\\cancer\\*.jpg'
    noncancer_train_path = 'D:\\oral cancer project\\train\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        train_data_knn.append(fd)
        train_labels_knn.append('cancer')

    for entry in glob.glob(noncancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        train_data_knn.append(fd)
        train_labels_knn.append('non-cancer')

    # Combine train data and labels for KNN
    train_data_knn = np.array(train_data_knn)
    train_labels_knn = np.array(train_labels_knn)

    # Split the data into train and validation sets for KNN
    train_data_knn, val_data_knn, train_labels_knn, val_labels_knn = train_test_split(
        train_data_knn, train_labels_knn, test_size=0.2, random_state=42)

    # Normalize the data for KNN
    scaler_knn = StandardScaler()
    train_data_knn = scaler_knn.fit_transform(train_data_knn)
    val_data_knn = scaler_knn.transform(val_data_knn)

    # Apply dimensionality reduction using PCA for KNN
    n_components_knn = min(train_data_knn.shape[0], train_data_knn.shape[1])
    pca_knn = PCA(n_components=n_components_knn)
    train_data_knn = pca_knn.fit_transform(train_data_knn)
    val_data_knn = pca_knn.transform(val_data_knn)

    # Load and preprocess the test images for KNN
    test_data_knn = []
    test_labels_knn = []

    cancer_test_path = 'D:\\oral cancer project\\test\\cancer\\*.jpg'
    noncancer_test_path = 'D:\\oral cancer project\\test\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        test_data_knn.append(fd)
        test_labels_knn.append('cancer')

    for entry in glob.glob(noncancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        test_data_knn.append(fd)
        test_labels_knn.append('non-cancer')

    # Convert test data and labels to arrays for KNN
    test_data_knn = np.array(test_data_knn)
    test_labels_knn = np.array(test_labels_knn)

    # Normalize the test data for KNN
    test_data_knn = scaler_knn.transform(test_data_knn)

    # Reduce dimensionality using PCA for KNN
    test_data_knn = pca_knn.transform(test_data_knn)

    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_data_knn, train_labels_knn)

    # Predict on the validation set
    val_predictions_knn = knn.predict(val_data_knn)

    # Evaluate the accuracy of the KNN classifier
    accuracy_knn = metrics.accuracy_score(val_labels_knn, val_predictions_knn)
    print("KNN Accuracy:", accuracy_knn)

    # Load and preprocess the images for the neural network
    train_data_nn = []
    train_labels_nn = []

    cancer_train_path = 'D:\\oral cancer project\\train\\cancer\\*.jpg'
    noncancer_train_path = 'D:\\oral cancer project\\train\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        train_data_nn.append(preprocessed_img)
        train_labels_nn.append(1)

    for entry in glob.glob(noncancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        train_data_nn.append(preprocessed_img)
        train_labels_nn.append(0)

    # Convert train data and labels to arrays for the neural network
    train_data_nn = np.array(train_data_nn)
    train_labels_nn = np.array(train_labels_nn)

    # Split the data into train and validation sets for the neural network
    train_data_nn, val_data_nn, train_labels_nn, val_labels_nn = train_test_split(
        train_data_nn, train_labels_nn, test_size=0.2, random_state=42)

    # Load the pre-trained feature extraction model
    feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,input_shape=(224, 224, 3))

    # Create a sequential model
    model = tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dense(1, activation='sigmoid')])

    # Compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # Preprocess the data for the neural network
    train_data_nn = tf.image.resize(train_data_nn, (224, 224))
    val_data_nn = tf.image.resize(val_data_nn, (224, 224))

    # Train the neural network
    history = model.fit(train_data_nn, train_labels_nn, epochs=10, validation_data=(val_data_nn, val_labels_nn))

    # Load and preprocess the test images for the neural network
    test_data_nn = []
    test_labels_nn = []

    cancer_test_path = 'D:\\oral cancer project\\test\\cancer\\*.jpg'
    noncancer_test_path = 'D:\\oral cancer project\\test\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        test_data_nn.append(preprocessed_img)
        test_labels_nn.append(1)

    for entry in glob.glob(noncancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        test_data_nn.append(preprocessed_img)
        test_labels_nn.append(0)

    # Convert test data and labels to arrays for the neural network
    test_data_nn = np.array(test_data_nn)
    test_labels_nn = np.array(test_labels_nn)

    # Preprocess the test data for the neural network
    test_data_nn = tf.image.resize(test_data_nn, (224, 224))

    # Evaluate the neural network on the test set
    test_loss, test_accuracy = model.evaluate(test_data_nn, test_labels_nn)
    print("Neural Network Accuracy:", test_accuracy)

    # Calculate predictions for the test set
    test_predictions_nn = model.predict(test_data_nn)
    test_predictions_nn = np.round(test_predictions_nn).flatten()

    # Calculate evaluation metrics for the neural network
    accuracy_nn = metrics.accuracy_score(test_labels_nn, test_predictions_nn)
    precision_nn = metrics.precision_score(test_labels_nn, test_predictions_nn)
    f1_score_nn = metrics.f1_score(test_labels_nn, test_predictions_nn)
    recall_nn = metrics.recall_score(test_labels_nn, test_predictions_nn)

    print("Neural Network Accuracy:", accuracy_nn)
    print("Neural Network Precision:", precision_nn)
    print("Neural Network F1 Score:", f1_score_nn)
    print("Neural Network Recall:", recall_nn)
    print("Neural Network Loss:", test_loss)

    # Plotting the graph
    def plot_metric(history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    # Plot accuracy
    plot_metric(history, 'accuracy')

    # Plot loss
    plot_metric(history, 'loss')

import tkinter as tk
from tkinter import filedialog, messagebox
from skimage.feature import hog
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub

# Function to extract HOG features from an image
def extract_hog_features(img):
    resized_img = cv2.resize(img, (64, 128))
    resized_img = cv2.convertScaleAbs(resized_img)  # Convert to 8-bit unsigned integer
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    fd = hog(gray_img, feature_vector=True)
    return fd

# Function to preprocess the image data
def preprocess_image(img):
    normalized_img = img.astype(float) / 255.0  # Normalize image
    return normalized_img

# Function to train and evaluate KNN classifier
def train_evaluate_knn():
    # Load and preprocess the training images for KNN
    train_data_knn = []
    train_labels_knn = []

    cancer_train_path = 'D:\\oral cancer project\\train\\cancer\\*.jpg'
    noncancer_train_path = 'D:\\oral cancer project\\train\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        train_data_knn.append(fd)
        train_labels_knn.append('cancer')

    for entry in glob.glob(noncancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        train_data_knn.append(fd)
        train_labels_knn.append('non-cancer')

    # Combine train data and labels for KNN
    train_data_knn = np.array(train_data_knn)
    train_labels_knn = np.array(train_labels_knn)

    # Split the data into train and validation sets for KNN
    train_data_knn, val_data_knn, train_labels_knn, val_labels_knn = train_test_split(
        train_data_knn, train_labels_knn, test_size=0.2, random_state=42)

    # Normalize the data for KNN
    scaler_knn = StandardScaler()
    train_data_knn = scaler_knn.fit_transform(train_data_knn)
    val_data_knn = scaler_knn.transform(val_data_knn)

    # Apply dimensionality reduction using PCA for KNN
    n_components_knn = min(train_data_knn.shape[0], train_data_knn.shape[1])
    pca_knn = PCA(n_components=n_components_knn)
    train_data_knn = pca_knn.fit_transform(train_data_knn)
    val_data_knn = pca_knn.transform(val_data_knn)

    # Load and preprocess the test images for KNN
    test_data_knn = []
    test_labels_knn = []

    cancer_test_path = 'D:\\oral cancer project\\test\\cancer\\*.jpg'
    noncancer_test_path = 'D:\\oral cancer project\\test\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        test_data_knn.append(fd)
        test_labels_knn.append('cancer')

    for entry in glob.glob(noncancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        fd = extract_hog_features(preprocessed_img)
        test_data_knn.append(fd)
        test_labels_knn.append('non-cancer')

    # Convert test data and labels to arrays for KNN
    test_data_knn = np.array(test_data_knn)
    test_labels_knn = np.array(test_labels_knn)

    # Normalize the test data for KNN
    test_data_knn = scaler_knn.transform(test_data_knn)

    # Reduce dimensionality using PCA for KNN
    test_data_knn = pca_knn.transform(test_data_knn)

    # Train a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_data_knn, train_labels_knn)

    # Predict on the validation set
    val_predictions_knn = knn.predict(val_data_knn)

    # Evaluate the accuracy of the KNN classifier
    accuracy_knn = metrics.accuracy_score(val_labels_knn, val_predictions_knn)
    messagebox.showinfo("KNN Accuracy", f"Accuracy: {accuracy_knn}")

# Function to train and evaluate the neural network
def train_evaluate_neural_network():
    # Load and preprocess the training images for the neural network
    train_data_nn = []
    train_labels_nn = []

    cancer_train_path = 'D:\\oral cancer project\\train\\cancer\\*.jpg'
    noncancer_train_path = 'D:\\oral cancer project\\train\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        train_data_nn.append(preprocessed_img)
        train_labels_nn.append(1)

    for entry in glob.glob(noncancer_train_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        train_data_nn.append(preprocessed_img)
        train_labels_nn.append(0)

    # Convert train data and labels to arrays for the neural network
    train_data_nn = np.array(train_data_nn)
    train_labels_nn = np.array(train_labels_nn)

    # Split the data into train and validation sets for the neural network
    train_data_nn, val_data_nn, train_labels_nn, val_labels_nn = train_test_split(
        train_data_nn, train_labels_nn, test_size=0.2, random_state=42)

    # Load the pre-trained feature extraction model
    feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url, input_shape=(224, 224, 3))

    # Create a sequential model
    model = tf.keras.Sequential([feature_extractor_layer, tf.keras.layers.Dense(1, activation='sigmoid')])

    # Compile the model
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # Preprocess the data for the neural network
    train_data_nn = tf.image.resize(train_data_nn, (224, 224))
    val_data_nn = tf.image.resize(val_data_nn, (224, 224))

    # Train the neural network
    history = model.fit(train_data_nn, train_labels_nn, epochs=10, validation_data=(val_data_nn, val_labels_nn))

    # Load and preprocess the test images for the neural network
    test_data_nn = []
    test_labels_nn = []

    cancer_test_path = 'D:\\oral cancer project\\test\\cancer\\*.jpg'
    noncancer_test_path = 'D:\\oral cancer project\\test\\non-cancer\\*.jpg'

    for entry in glob.glob(cancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        test_data_nn.append(preprocessed_img)
        test_labels_nn.append(1)

    for entry in glob.glob(noncancer_test_path):
        img = mpimg.imread(entry)
        preprocessed_img = preprocess_image(img)
        test_data_nn.append(preprocessed_img)
        test_labels_nn.append(0)

    # Convert test data and labels to arrays for the neural network
    test_data_nn = np.array(test_data_nn)
    test_labels_nn = np.array(test_labels_nn)

    # Preprocess the test data for the neural network
    test_data_nn = tf.image.resize(test_data_nn, (224, 224))

    # Evaluate the neural network on the test set
    test_loss, test_accuracy = model.evaluate(test_data_nn, test_labels_nn)
    messagebox.showinfo("Neural Network Accuracy", f"Accuracy: {test_accuracy}")

    # Calculate predictions for the test set
    test_predictions_nn = model.predict(test_data_nn)
    test_predictions_nn = np.round(test_predictions_nn).flatten()

    # Calculate evaluation metrics for the neural network
    accuracy_nn = metrics.accuracy_score(test_labels_nn, test_predictions_nn)
    precision_nn = metrics.precision_score(test_labels_nn, test_predictions_nn)
    f1_score_nn = metrics.f1_score(test_labels_nn, test_predictions_nn)
    recall_nn = metrics.recall_score(test_labels_nn, test_predictions_nn)

    messagebox.showinfo("Neural Network Metrics",
                        f"Accuracy: {accuracy_nn}\nPrecision: {precision_nn}\nF1 Score: {f1_score_nn}\nRecall: {recall_nn}\nLoss: {test_loss}")

    # Plotting the graph
    def plot_metric(history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.title('Model ' + metric)
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    # Plot accuracy
    plot_metric(history, 'accuracy')

    # Plot loss
    plot_metric(history, 'loss')

# Create the GUI
root = tk.Tk()
root.title("Oral Cancer Detection")
root.geometry("400x300")

# Function to open the file dialog and select the folder
def open_folder_dialog():
    folder_path = filedialog.askdirectory()
    messagebox.showinfo("Folder Selected", f"Selected Folder: {folder_path}")

# Button to select the folder
folder_button = tk.Button(root, text="Select Folder", command=open_folder_dialog)
folder_button.pack(pady=20)

# Button to train and evaluate KNN classifier
knn_button = tk.Button(root, text="Train & Evaluate KNN", command=train_evaluate_knn)
knn_button.pack(pady=10)

# Button to train and evaluate the neural network
nn_button = tk.Button(root, text="Train & Evaluate Neural Network", command=train_evaluate_neural_network)
nn_button.pack(pady=10)

root.mainloop()
