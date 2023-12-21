import tkinter as tk
r=tk.Tk()
r.title('Counting seconds')
button=tk.Button(r,text='Stop',width=25,command=r.destroy)
button.grid()
#button.pick()
r.mainloop()

#%%

from tkinter import *
master=Tk()
w=Canvas(master,width=40,height=60)
w.pack()
canvas_height=20
canvas_width=200
y=int(canvas_height/2)
w.create_line(0,y,canvas_width,100)
#w.create_line(0,y,canvas_width,y)
mainloop()

#%%

from tkinter import *
root=Tk()
T=Text(root,height=2,width=30)
T.pack()
T.insert(END,'DDU\nDHARMSIMH DESAI UNIVERSITY')
mainloop()

#%%

from tkinter import *
top=Tk()
Lb=Listbox(top)
Lb.insert(1,'Python')
Lb.insert(2,'Java')
Lb.insert(3,'C++')
Lb.insert(4,'Any other')
Lb.pack()
top.mainloop()

#%%

from tkinter import *
def submit():
    first_name=a1.get()
    last_name=a2.get()
    print("First name:",first_name)
    print("Last name:",last_name)
    master.destroy()
master=Tk()
Label(master,text='First name').grid(row=0)
Label(master,text='Last Name').grid(row=1)
a1=Entry(master)
a2=Entry(master)
a1.grid(row=0,column=1)
a2.grid(row=1,column=1)
w=Button(master,text='Enter',width=20,command=submit)
w.grid(row=2,column=1)
mainloop()
