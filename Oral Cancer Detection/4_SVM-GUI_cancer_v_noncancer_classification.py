from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox
import os
import numpy as np
import cv2
import pickle
from sklearn.svm import SVC


# Load the SVM model
pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()


# Function to perform image classification
def classify_image(image_path):
    # Load and preprocess the image
    oc_img = cv2.imread(image_path, 0)
    oc_img = cv2.resize(oc_img, (50, 50))
    image = np.array(oc_img).flatten()

    # Perform prediction using the SVM model
    prediction = model.predict([image])

    # Display the result
    result_label.configure(text="Cancerous" if prediction == 1 else "Non-Cancerous")


# Function to handle image upload and classification
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("JPEG files", "*.jpg")])
    if file_path:
        classify_image(file_path)
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        image = ImageTk.PhotoImage(image)
        image_label.configure(image=image)
        image_label.image = image  


def login():
    username1 = username.get()
    password1 = password.get()
    if username1 == "admin" and password1 == "password":
        messagebox.showinfo("Login", "Login Successful")
        root.withdraw()  # Hide the login window
        image_window.deiconify()  # Open the image upload window
    else:
        messagebox.showerror("Fail", "Invalid Username or Password")


# Create the login window
root = Tk()
root.title("Login")
root.geometry("600x400")
root.minsize(200, 200)
root.maxsize(300, 700)
root.configure(bg="lightblue")
root.iconbitmap("D:\\oral cancer project\\GUI\\logo.ico")

# Create the image upload window
image_window = Toplevel(root)
image_window.title("Oral Cancer Detection")
image_window.withdraw()
image_window.geometry("400x400")

# Create the GUI widgets for the login window
image = Image.open("D:\\oral cancer project\\GUI\\ORAL CANCER PREDICTION.jpg")
resized_image = image.resize((100, 100))
photo = ImageTk.PhotoImage(resized_image)
label = Label(root, image=photo)
label.pack(pady=10)

label = Label(root, text="Enter Username", bg="lightblue", foreground="black", font=("Arial", 15, "bold"))
label.pack()
username = Entry(root, width=20, font=("Arial", 14), bg="gray", fg="white", borderwidth=3)
username.pack(pady=5)

label = Label(root, text="Enter Password", bg="lightblue", foreground="black", font=("Arial", 15, "bold"))
label.pack()
password = Entry(root, width=20, font=("Arial", 14), bg="gray", fg="white", borderwidth=3, show="*")
password.pack(pady=5)

login_button = Button(root, text="Login", bg="lightgreen", activebackground="cyan", borderwidth=3,
                      font=("Arial", 11, "bold"), command=login)
login_button.pack(pady=10)


# Create the GUI widgets for the image upload window
upload_button = Button(image_window, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

image_label = Label(image_window)
image_label.pack()

result_label = Label(image_window, text="")
result_label.pack(pady=10)


root.mainloop()
