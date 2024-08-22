import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
from sklearn import svm

# Load the SVM model
model = svm.SVC()  # Replace with your trained SVM model

# Create the GUI window
window = tk.Tk()
window.title("Oral Cancer Detection")
window.geometry("400x400")

# Define the function to handle image classification
def classify_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).resize((64, 64))
    image_array = np.array(image)  # Convert image to numpy array
    flattened_image = image_array.flatten().reshape(1, -1)  # Flatten the image array

    # Perform prediction using the SVM model
    prediction = model.predict(flattened_image)

    # Display the result
    result_label.configure(text="Cancerous" if prediction == 1 else "Non-Cancerous")

# Define the function to handle image upload and classification
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg")])
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image  # Store the image reference

        classify_image(file_path)  # Call the classification function

# Create GUI widgets
upload_button = tk.Button(window, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

image_label = tk.Label(window)
image_label.pack()

result_label = tk.Label(window, text="")
result_label.pack(pady=10)

# Start the GUI event loop
window.mainloop()
