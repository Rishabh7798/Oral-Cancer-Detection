from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox

root = Tk()

root.title("Login")

root.geometry("600x400")

root.minsize(200,200)

root.maxsize(300,700)

root.iconbitmap("D:\\oral cancer project\\GUI\\logo.ico")

root.configure(bg="lightblue")

# logo on page
image = Image.open("D:\\oral cancer project\\GUI\\ORAL CANCER PREDICTION.jpg")

resizedImage = image.resize((100,100))

photo = ImageTk.PhotoImage(resizedImage)

label = Label(root, image=photo)
label.pack(pady=10)


label=Label(root, text="Enter Username", bg = "lightblue", foreground="black",font=("Arial",15,"bold"))
label.pack()

username=Entry(root, width = 20, font=("Arial",14), bg="gray",fg="white", borderwidth=3)
username.pack(pady=5)

label=Label(root, text="Enter Password", bg = "lightblue", foreground="black",font=("Arial",15,"bold"))
label.pack()

password=Entry(root, width = 20, font=("Arial",14), bg="gray",fg="white", borderwidth=3, show="*")
password.pack(pady=5)

def login():
    username1=username.get()
    password1=password.get()
    if username1=="admin" and password1=="password":
        messagebox.showinfo("Login","Login Successful")
    else:
        messagebox.showerror("Fail", "Invalid Username or Password")



loginButton=Button(root, text="Login",bg="lightgreen", activebackground="cyan",borderwidth=3, font=("Arial", 11,"bold"),command=login)
loginButton.pack(pady=10)




root.mainloop()