# Image Classification using Transfer Learning

This project demonstrates image classification using transfer learning with a pre-trained MobileNet V2 model. The dataset used for training and testing is the "Dogs vs Cats" dataset from Kaggle.

## Table of Contents
- [Introduction](#introduction)
- [Workflow](#workflow)
- [Setup Instructions](#setup-instructions)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Dependencies](#dependencies)

## Introduction

Transfer learning is a deep learning technique where a pre-trained model is used for a similar task with a smaller dataset. This approach often results in higher accuracy compared to training models from scratch.

### Pre-trained Models Used
1. VGG-16
2. ResNet50
3. Inceptionv3
4. MobileNet V2 (used in this project)

## Workflow

1. Dataset acquisition
2. Image processing
3. Train-test split
4. Using pre-trained MobileNet V2 model
5. Training MobileNet model on the "Dogs vs Cats" dataset

## Setup Instructions

1. Search for "cats vs dogs" on Google.
2. Open Kaggle and go to account settings to create an API token (a JSON file will be downloaded).
3. Upload the token file (`kaggle.json`) to your Colab environment.
4. Accept the competition rules on Kaggle.

## Data Preparation

1. Install the Kaggle library.
    ```python
    !pip install Kaggle
    ```

2. Configure the path of the Kaggle JSON file.
    ```python
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json
    ```

3. Download the "Dogs vs Cats" dataset from Kaggle.
    ```python
    !kaggle competitions download -c dogs-vs-cats
    ```

4. Extract the dataset.
    ```python
    from zipfile import ZipFile

    dataset = '/content/dogs-vs-cats.zip'
    with ZipFile(dataset, 'r') as zip:
        zip.extractall()
        print('Dataset is extracted')
    ```

5. Count the number of images in the training folder.
    ```python
    import os

    path, dirs, files = next(os.walk('/content/train'))
    file_count = len(files)
    print("Number of Images: ", file_count)
    ```

6. Resize all images to (224, 224) for use with MobileNet V2.
    ```python
    from PIL import Image

    original_folder = '/content/train/'
    resized_folder = '/content/image_resized/'
    os.mkdir(resized_folder)

    for i in range(2000):
        filename = os.listdir(original_folder)[i]
        img_path = original_folder + filename
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        newImgPath = resized_folder + filename
        img.save(newImgPath)
    ```

7. Assign labels (0 for cats, 1 for dogs) to the images.
    ```python
    filenames = os.listdir('/content/image_resized')
    labels = []

    for i in range(2000):
        file_name = filenames[i]
        first_char = file_name[0]
        labels.append(1 if first_char == 'd' else 0)
    ```

## Model Training

1. Convert resized images to numpy arrays and split the data into training and testing sets.
    ```python
    import numpy as np
    import cv2
    import glob
    from sklearn.model_selection import train_test_split

    image_directory = '/content/image_resized/'
    image_extension = ['png', 'jpg']
    files = [glob.glob(image_directory + '*.' + e) for e in image_extension]
    dog_cat_images = np.asarray([cv2.imread(file) for file in files])
    x = dog_cat_images
    y = np.asarray(labels)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
    x_train_scaled = x_train / 255
    x_test_scaled = x_test / 255
    ```

2. Build and compile the neural network with the pre-trained MobileNet V2 model.
    ```python
    import tensorflow as tf
    import tensorflow_hub as hub

    mobilenet_model = 'https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4'
    pretrained_model = hub.KerasLayer(mobilenet_model, input_shape=(224,224,3), trainable=False)

    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.Dense(2)  # Output layer with 2 classes
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc']
    )
    model.summary()
    ```

3. Train the model.
    ```python
    model.fit(x_train_scaled, y_train, epochs=5)
    ```

## Evaluation

Evaluate the model on the testing set.
```python
score, acc = model.evaluate(x_test_scaled, y_test)
print("Loss:", score)
print("Test Accuracy:", acc)
```

## Prediction

Predict the class of a new image.

```python
input_image_path = '/content/download.jpg'
input_image = cv2.imread(input_image_path)
cv2_imshow(input_image)

input_image_resize = cv2.resize(input_image, (224, 224))
input_image_scaled = input_image_resize / 255
image_reshaped = np.reshape(input_image_scaled, [1, 224, 224, 3])

input_prediction = model.predict(image_reshaped)
input_prediction_label = np.argmax(input_prediction)

if input_prediction_label == 0:
    print("The image is of a CAT")
else:
    print("The image is of a DOG")
```

## Dependencies

1. TensorFlow
2. TensorFlow Hub
3. NumPy
4. PIL (Python Imaging Library)
5. OpenCV
6. Matplotlib
7. Scikit-learn
8. Kaggle API

To install the dependencies, run:

```python
!pip install tensorflow tensorflow_hub numpy pillow opencv-python matplotlib scikit-learn kaggle
```
