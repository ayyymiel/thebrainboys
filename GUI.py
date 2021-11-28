from tkinter import *
import LivePredictCorrelation as Predict
import serial


class ArduinoBot:
    def __init__(self):
        self.ser = serial.Serial("COM4", 9600, timeout=1)

    def move_b(self):
        self.ser.write(b'b')

    def move_f(self):
        self.ser.write(b'f')

    def move_l(self):
        self.ser.write(b'l')

    def move_r(self):
        self.ser.write(b'r')


def board_connection():
    Predict.OpenBCI()


root = Tk()
root.geometry('600x300')
f1 = Frame(root)
f1.grid(row=0, column=0)
l = Label(f1, text="The Brain Boys")
l.grid(row=0, column=0)

# Connect to Cyton
b = Button(f1, text="Connect", command=board_connection)
b.grid(row=1, column=0, padx=20, pady=10)
b = Button(f1, text="Predict", command=Predict.OpenBCI.predict)
b.grid(row=1, column=1)
f2 = Frame(root)

# Connect with Arduino Bot
f2.grid(row=2, column=0, pady=20)
l = Label(f2, text="Bot Control")
l.grid(row=0, column=0, pady=5)
b = Button(f2, text="Connect", width=10, command=ArduinoBot)
b.grid(row=1, column=0)
b = Button(f2, text="Left", width=10, command=ArduinoBot.move_l)
b.grid(row=2, column=0)
b = Button(f2, text="Right", width=10, command=ArduinoBot.move_r)
b.grid(row=3, column=0)
b = Button(f2, text="Backward", width=10, command=ArduinoBot.move_b)
b.grid(row=4, column=0)
b = Button(f2, text="Forward", width=10, command=ArduinoBot.move_f)
b.grid(row=5, column=0)
f3 = Frame(root)
f3.grid(row=1, rowspan=5, column=2)
t = Text(f3, height=7, width=40)
t.grid(row=1, column=3)


root.mainloop()
