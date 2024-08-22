from tkinter import *
from tkinter import messagebox

# constructor
root = Tk()

root.title("Oral Cancer")

root.geometry("400x300")

root.minsize(200,200)

root.maxsize(600,600)

root.iconbitmap("GUI\\logo.ico")

root.configure(bg="lightblue")

#importing widgets
label = Label(root, text="Oral Cancer", font=("Arial",20,"bold"), width=15, height=5, bg="yellow", foreground="blue")
# label.pack(side=RIGHT)
label.pack()

# text typed by user
myEntry1=Entry(root, width=30, font=("Arial",14), bg="gray", fg="white", borderwidth=3)
myEntry1.pack()
# if we want to set this entry as password 
myEntry2=Entry(root, width=30, font=("Arial",14), bg="gray", fg="white", borderwidth=3, show="*")
myEntry2.pack() 

# to show button click message
def test():
    messagebox.showinfo("Info","Button Clicked")
    # messagebox.showerror("Error","Button Clicked")
    # messagebox.showwarning("Warning","Button Clicked")


# button method 
myButton=Button(text="click me!!!", command=test, bg="pink", borderwidth=3, activebackground="blue")
# creating space between myEntry and myButton
myButton.pack(pady=10)

# OptionMenu
options = ["Option1", "Option2", "Option3"]
selectedOption = StringVar()
selectedOption.set(options[0])

optionMenu=OptionMenu(root, selectedOption, *options)
optionMenu.configure(bg="green", fg="white", font=("Arial",12),width=15)
optionMenu["menu"].configure(bg="green", fg="white", font = ("Arial", 12))
optionMenu.pack()

# starts the gui of tkinter program and keeps it running
root.mainloop()