import pandas as pd
import matplotlib.pyplot as plt
import os
import csv

labels = []

dir = 'captures'

if __name__ == "__main__":

    for file in os.listdir(dir):
        key = file.rsplit(('.'), 1)[0].rsplit(' ')[1]
        if key == 'n':
            labels.append({"filename": file, "class": 0})
        elif key == 'space':
            labels.append({"filename": file, "class": 1})

    field_names = ['filename', 'class']

    with open('labels_dino.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(labels)
