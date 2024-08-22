import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import threading
import os

# Create the main application window
window = tk.Tk()
window.title("Oral Cancer Detection")

# Create a global variable to store the uploaded image
uploaded_image = None

# Define functions for each page
def show_login_page():
    # Clear the window
    clear_window()
    
    
    # Create login page elements
    label = tk.Label(window, text="Oral Cancer Detection", font=("Arial", 18))
    label.pack(pady=20)
    
    username_label = tk.Label(window, text="Username:")
    username_label.pack()
    
    username_entry = tk.Entry(window)
    username_entry.pack()

    password_label = tk.Label(window, text="\nPassword:")
    password_label.pack()
    
    password_entry = tk.Entry(window, show="*")
    password_entry.pack()
    
    # Function to handle login button click
    def login():
        # Get the entered username and password
        username = username_entry.get()
        password = password_entry.get()
        
        # If login is successful, proceed to the upload page
        if username == "admin" and password == "password":
            show_upload_page()
        else:
            # Display an error message for invalid credentials
            error_label = tk.Label(window, text="Invalid username or password", fg="red")
            error_label.pack(pady=10)
    
    # Create a login button
    login_button = tk.Button(window, text="Login", command=login)
    login_button.pack(pady=10)


def show_upload_page():
    # Clear the window
    clear_window()
    
    # Create upload page elements
    label = tk.Label(window, text="Upload an Image", font=("Arial", 18))
    label.pack(pady=20)
    
    # Function to handle image upload
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if file_path:
            # Open the image file
            image = Image.open(file_path)
            
            # Resize the image to fit the display
            image = image.resize((300, 300))
            
            # Convert the image to Tkinter format
            global uploaded_image
            uploaded_image = ImageTk.PhotoImage(image)
            
            # Display the uploaded image
            image_label = tk.Label(window, image=uploaded_image)
            image_label.pack(pady=20)
            
            # Call the next page function (e.g., analyze_page)
            show_analyze_page()
    
    # Create an "Upload" button
    upload_button = tk.Button(window, text="Upload Image", command=upload_image)
    upload_button.pack(pady=10)

def show_analyze_page():
    # Clear the window
    clear_window()

    # Create analyze page elements
    label = tk.Label(window, text="Analyzing...", font=("Arial", 18))
    label.pack(pady=20)

    # Function to analyze the uploaded image
    def analyze_image():
        analyzing_label = tk.Label(window, text="Analyzing...", font=("Arial", 18))
        analyzing_label.pack(pady=20)

        # Simulate image classification result (0: Non-Cancer, 1: Cancer)
        classification_result = 1

        show_result_page(classification_result)

    # Start analyzing the image in a separate thread
    analyze_thread = threading.Thread(target=analyze_image)
    analyze_thread.start()

def show_result_page(classification_result):
    # Clear the window
    clear_window()

    # Create result page elements
    label = tk.Label(window, text="Result", font=("Arial", 18))
    label.pack(pady=20)

    # Display the result (e.g., cancer or non-cancer) and the percentage of chances
    result_label = tk.Label(window, text="Classification Result: ", font=("Arial", 16))
    result_label.pack(pady=10)

    if classification_result == 0:
        result_text = "Non-Cancer"
    else:
        result_text = "Cancer"

    result_value_label = tk.Label(window, text=result_text, font=("Arial", 16))
    result_value_label.pack(pady=10)


def clear_window():
    # Clear all elements from the window
    for widget in window.winfo_children():
        widget.destroy()

# Show the login page by default
show_login_page()

# Start the main event loop
window.mainloop()
