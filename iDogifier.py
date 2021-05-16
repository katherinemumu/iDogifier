import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import tkinter as tk
import tkinter.font as font

window = tk.Tk()
window.title("iDogifier")

#Prompt user input for image file
title = tk.Label(text="Image file name:")
userInput = tk.Entry()
title.pack()
userInput.pack()

#Store image file name
imageFile = userInput.get()

#Click on button to start comparison
font = font.Font(family='Lucida handwriting', size = 15, weight = "bold")
compareButton = tk.Button(
    text = "Click to start!",
    width=25,
    height=3,
    bg="#FCEDFC",
    fg="black"
)
compareButton["font"] = font
compareButton.pack()

def handle_click(event):
    imageFile = userInput.get()
    batch_size = 32
    img_height = 180
    img_width = 180
    class_names = ['Husky', 'Pomeranian', 'Pug']

    new_model = tf.keras.models.load_model('dogmodel')

    dir_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(dir_path, imageFile)

    img = keras.preprocessing.image.load_img(
        img_path, target_size=(img_height, img_width)
    )

    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = new_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    #Output text results and image
    outcome = tk.Label(
        text="This image most likely belongs to {} with a {:.2f}% confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)),
        foreground= "#150052",
        background= "#9AE4FF",
        width=100,
        height=50
    )
    outcome["font"] = font

    outcome.pack()

compareButton.bind("<Button-1>", handle_click)

window.mainloop()


