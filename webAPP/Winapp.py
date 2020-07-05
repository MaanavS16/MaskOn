import tkinter as tk
import tensorflow as tf
import pickle as pkl
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import *
from matplotlib.figure import Figure
from modelModule import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

testObj = cnnTF()
root = tk.Tk()
root.wm_title("Mask On")
# fig = Figure(figsize=(4, 4), dpi=140)
# ax = fig.add_subplot(1,1,1)

# canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
# canvas.get_tk_widget().grid(row=0, column=0)
# canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=0)
plt.style.use('seaborn-pastel')
#w = tk.Scale(root, from_=1, to=10, orient = tk.HORIZONTAL, label = "Time")
#w.pack()
v = tk.StringVar()
c = tk.Label(root, textvariable = v).pack()
RECORD_SECONDS = 1
def Start():
    testObj.launchFramePreds()


def End():
    # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()



a = tk.Button(root, text ="StartRecording", command = Start)
a.pack()
b = tk.Button(root, text ="StopRecording", command = End)
b.pack()

root.mainloop()
