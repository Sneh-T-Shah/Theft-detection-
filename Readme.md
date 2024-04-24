
# Theft Detection Model

This project is aimed at building a machine learning model for detecting shoplifting incidents using video data from the DCSASS Dataset. The model is designed to classify video frames as either containing a shoplifting event or not.

## Dataset

The DCSASS Dataset contains video footage of shoplifting incidents, along with corresponding labels. The dataset is available on Kaggle: [DCSASS Dataset](https://www.kaggle.com/datasets/dcsass-dataset).

## Preprocessing

The preprocessing steps involve the following:

1. Loading the dataset and preparing the file paths for the video files.
2. Creating directories to store the extracted frames from the videos.
3. Extracting individual frames from the video files and saving them in the respective directories (0 for non-shoplifting and 1 for shoplifting).
4. Ensuring that the number of frames for both classes is approximately balanced.

## Data Generator

An `ImageDataGenerator` from TensorFlow's `keras.preprocessing.image` module is used to create data generators for the training and validation sets. This allows efficient loading and preprocessing of the image data during the training process.

## Model Architecture

The model uses transfer learning with the MobileNetV2 architecture as the base model. The top layers of the MobileNetV2 model are removed, and new dense layers are added for binary classification. The architecture of the model is as follows:

1. MobileNetV2 base model (without top layers)
2. Flatten layer
3. Dense layer (512 units, ReLU activation)
4. Dense layer (512 units, Sigmoid activation)
5. Dropout layer (0.5 dropout rate)
6. Dense output layer (1 unit, Sigmoid activation)

## Training

The model is trained using the `fit` method from TensorFlow's `keras.models.Sequential` class. The training process includes the following:

- Optimization algorithm: Nadam with a weight decay of 0.0001
- Loss function: Binary cross-entropy
- Metrics: Accuracy
- Callbacks: ReduceLROnPlateau (to reduce the learning rate if the validation loss doesn't improve) and ModelCheckpoint (to save the best model weights based on validation accuracy)

The model is trained for 10 epochs with a batch size of 32.

## Evaluation

During training, the model's performance is evaluated on both the training and validation sets. The best model weights, based on the highest validation accuracy, are automatically saved using the `ModelCheckpoint` callback.

After training, the model is evaluated on the validation set, and the following metrics are calculated:

- Overall Accuracy: 90.61%
- Recall: 90.61%
- F1 Score: 90.67%
- Confusion Matrix:
  ```
  [[104  11]
   [  6  60]]
  ```
  
- Classification Report:
  ```
                precision    recall  f1-score   support

             0       0.95      0.90      0.92       115
             1       0.85      0.91      0.88        66

    accuracy                           0.91       181
   macro avg       0.90      0.91      0.90       181
weighted avg       0.91      0.91      0.91       181
  ```

## Usage

To use the trained model for inference, load the best model weights and pass in new video frames or sequences for classification.

