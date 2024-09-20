import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = load_img(os.path.join(folder, filename), target_size=(64, 64))
        if img is not None:
            images.append(img_to_array(img))
    return np.array(images)

# GUI to select folders
root = tk.Tk()
root.withdraw()
folder_0 = filedialog.askdirectory(title='Select folder with images labeled as 0')
folder_1 = filedialog.askdirectory(title='Select folder with images labeled as 1')

# Load images
images_0 = load_images_from_folder(folder_0)
images_1 = load_images_from_folder(folder_1)

# Create labels
labels_0 = np.zeros(len(images_0))
labels_1 = np.ones(len(images_1))

# Combine and preprocess data
X = np.concatenate((images_0, images_1), axis=0)
y = np.concatenate((labels_0, labels_1), axis=0)
X = X / 255.0  # Normalize images

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the model for use with scikit-learn
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters for GridSearchCV
param_grid = {
    'batch_size': [10, 20],
    'epochs': [10, 20],
    'optimizer': ['adam', 'rmsprop']
}

# Perform GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Evaluate the model
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Best Accuracy: {accuracy}')
print(f'Best Hyperparameters: {grid_result.best_params_}')
