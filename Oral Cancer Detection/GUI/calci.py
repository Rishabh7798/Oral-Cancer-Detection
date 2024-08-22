from tkinter import *
from PIL import ImageTk, Image
from tkinter import messagebox

root = Tk()

root.title("Calculator")

root.geometry("600x400")

root.minsize(200,200)

root.maxsize(300,700)

root.iconbitmap("D:\\oral cancer project\\GUI\\logo.ico")

root.configure(bg="lightblue")

'''
# logo on page
image = Image.open("")

resizedImage = image.resized((100,100))

photo = ImageTk.PhotoImage(resizedImage)

label = Label(root, image=photo)
label.pack(pady=10)

'''
label=Label(root, text="Enter First Number", bg = "lightblue", foreground="black",font=("Arial",15,"bold"))
label.pack()

firstNum=Entry(root, width = 15, font=("Arial",14), bg="gray",fg="white", borderwidth=3)
firstNum.pack(pady=5)

label=Label(root, text="Enter Second Number", bg = "lightblue", foreground="black",font=("Arial",15,"bold"))
label.pack()

secondNum=Entry(root, width = 15, font=("Arial",14), bg="gray",fg="white", borderwidth=3)
secondNum.pack(pady=5)

result_label=Label(root, text="Result", bg = "lightblue", foreground="black",font=("Arial",15,"bold"))
result_label.pack()
def calci():
    num1=float(firstNum.get())
    num2=float(secondNum.get())
    
    result=num1+num2
    result_label.config(text="Result is "+str(result))



loginButton=Button(root, text="Add",bg="lightgreen", activebackground="cyan",borderwidth=3, font=("Arial", 11,"bold"),command=calci)
loginButton.pack(pady=10)


root.mainloop()