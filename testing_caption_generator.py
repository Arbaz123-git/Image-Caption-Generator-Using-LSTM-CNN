from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
# standard library module in Python for parsing command-line arguments. It allows you to easily write
#  user-friendly command-line interfaces for your scripts or programs.

ap = argparse.ArgumentParser()
# This line creates an ArgumentParser object named ap. The ArgumentParser class in the argparse module 
# is used to define and parse command-line arguments.

ap.add_argument('-i', '--image', required=True, help="Image Path")
# This line adds a command-line argument to the parser:
# -i is the short form of the argument.
# --image is the long form of the argument.
# required=True specifies that this argument must be provided by the user.
# help="Image Path" provides a description of the argument that will be displayed when the user asks for help.

args = vars(ap.parse_args())
# This line parses the command-line arguments provided by the user and stores them in the args variable.
#  The parse_args() method parses the command-line arguments, and vars() converts the resulting Namespace
#  object to a dictionary.

img_path = args['image']
# This line extracts the value associated with the 'image' key from the args dictionary and assigns it to the
#  variable img_path. This value represents the path to the image file provided by the user as a command-line argument. 

# img_path = '/content/drive/MyDrive/ML/Flickr8k_Dataset/Flicker8k_Dataset/111537222_07e56d5a30.jpg'


def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

# Try-Except Block:

# The function begins with a try-except block. It attempts to open the image file specified by the filename parameter using the Image.open() method from the PIL (Python Imaging Library) module. If there's an exception (e.g., if the file doesn't exist or is not a valid image file), it prints an error message.
# Image Preprocessing:

# If the image is successfully opened, it is resized to (299, 299) pixels. This is a common preprocessing step, especially when working with pre-trained image classification models like Xception.
# The image is then converted to a NumPy array using np.array(image). The array representation is necessary for further processing.
# Handling 4-Channel Images:

# If the image has four channels (RGBA), the code keeps only the first three channels, converting it to a standard three-channel (RGB) image.
# Array Manipulation:

# The image array is expanded along the first axis, adding an extra dimension to make it suitable for input to the neural network model.
# Normalization:

# The pixel values are normalized by dividing by 127.5 and then subtracting 1.0. This normalization is often applied to bring the pixel values into a range suitable for the model.
# Feature Extraction:

# The pre-trained model (model) is then used to predict features from the preprocessed image. The extracted features are returned by the function.

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None

# Function Purpose:
# This function is designed to retrieve the word associated with a given integer index in a tokenizer.
# Parameters:

# integer: The integer index for which you want to find the corresponding word.
# tokenizer: The tokenizer object that contains the mapping between words and their integer indices.
# Iteration through Tokenizer Items:

# The function iterates through each item in the word_index attribute of the tokenizer. This attribute is a dictionary where words are keys, and their corresponding integer indices are values.
# Checking for Matching Index:

# For each iteration, the function checks if the integer index in the current iteration matches the input integer.
# Returning the Word:

# If a match is found, the function returns the corresponding word.
# If no match is found after iterating through all items, the function returns None.


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

# Parameters:

# model: The trained model used for generating descriptions.
# tokenizer: The tokenizer used for converting text to sequences and vice versa.
# photo: The input image features used as a seed for generating the description.
# max_length: The maximum length of the generated description.
# Initialization:

# The variable in_text is initialized with the string 'start'. This serves as the initial seed text for generating the description.
# Generating Text Sequence:

# The function enters a loop that iterates for a maximum of max_length times.
# Inside the loop:
# The current value of in_text is converted to a sequence of integers using the tokenizer (tokenizer.texts_to_sequences).
# The sequence is padded to match the maximum length (pad_sequences).
# The model is used to predict the next word given the input photo and the current sequence.
# The predicted word is determined as the word with the highest probability (argmax of the prediction).
# The predicted word is retrieved using the word_for_id function.
# The predicted word is appended to the in_text sequence.
# If the predicted word is 'end', the loop is terminated.
# Returning the Generated Text:

# The function returns the generated text sequence.


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
max_length = 32
tokenizer = load(open("/content/drive/MyDrive/ML/tokenizer.p","rb"))
model = load_model('/content/drive/MyDrive/ML/models6/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)

# Setting Parameters:

# max_length: The maximum length of the generated description sequence.
# tokenizer: Loading the tokenizer from a saved file using load from the pickle module.
# model: Loading the pre-trained image captioning model from a saved file using load_model from Keras.
# xception_model: Creating an instance of the Xception model for feature extraction.
# Extracting Features:

# img_path is assumed to be the path to an image file.
# photo is obtained by extracting features from the image using the Xception model (extract_features function).
# Loading Image:

# img is the image loaded using Image.open from the PIL module.
# Generating Description:

# generate_desc function is called with the loaded model, tokenizer, extracted photo features, and the maximum length.
# The generated description is stored in the variable description.
# Printing and Displaying Image:

# Two newline characters (\n\n) are printed for separation.
# The generated description is printed.
# The image is displayed using plt.imshow.