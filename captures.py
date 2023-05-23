import cv2
from PIL import ImageGrab
import numpy as np
import keyboard
import os
from datetime import datetime


current_key = ""
buffer = []

# check if folder named 'captures' exists. If not, create it.
if not os.path.exists("captures"):
    os.mkdir("captures")


def keyboardCallBack(key: keyboard.KeyboardEvent):
    global current_key

    if key.event_type == "down" and key.name not in buffer:
        buffer.append(key.name)

    if key.event_type == "up":
        buffer.remove(key.name)

    buffer.sort()
    current_key = " ".join(buffer)


if __name__ == "__main__":

    keyboard.hook(callback=keyboardCallBack)
    i = 0

    while (not keyboard.is_pressed("esc")):

        image = cv2.cvtColor(np.array(ImageGrab.grab(
            bbox=(455, 260, 1055, 380))), cv2.COLOR_RGB2BGR)  # Adapt the bbox to your screen
        if len(buffer) != 0:
            cv2.imwrite("captures/" + str(datetime.now()).replace("-", "_").replace(":",
                        "_").replace(" ", "_")+" " + current_key + ".png", image)
        else:
            cv2.imwrite("captures/" + str(datetime.now()).replace("-",
                        "_").replace(":", "_").replace(" ", "_") + " n" + ".png", image)
        i = i+1
