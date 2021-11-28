from tkinter import *
import LivePredictCorrelation as Predict
import serial


class ArduinoBot:
    def __init__(self):
        self.ser = serial.Serial("COM4", 9600, timeout=1)

    def movement(self, decision=int):
        if decision == 0:
            self.ser.write(b'b')
        elif decision == 1:
            self.ser.write(b'f')
        elif decision == 2:
            self.ser.write(b'l')
        elif decision == 4:
            self.ser.write(b'r')
        else:
            self.ser.write(b'o')

def start_gui():
    root = Tk()
    root.geometry('600x300')
    f1 = Frame(root)
    f1.grid(row=0, column=0)
    l = Label(f1, text="The Brain Boys")
    l.grid(row=0, column=0)

    # Connect to Cyton
    b = Button(f1, text="Connect", command=Predict.OpenBCI)
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
    b = Button(f2, text="Left", width=10, command=ArduinoBot.movement(decision=2))
    b.grid(row=2, column=0)
    b = Button(f2, text="Right", width=10, command=ArduinoBot.movement(decision=4))
    b.grid(row=3, column=0)
    b = Button(f2, text="Backward", width=10)
    b.grid(row=4, column=0)
    b = Button(f2, text="Forward", width=10)
    b.grid(row=5, column=0)
    f3 = Frame(root)
    f3.grid(row=1, rowspan=5, column=2)
    t = Text(f3, height=7, width=40)
    t.grid(row=1, column=3)

    root.mainloop()


start_gui()
