# MNIST Dropout Regularization Experiment

This repository contains a CS478 deep learning notebook that evaluates dropout regularization on a multilayer perceptron trained with the MNIST handwritten digit dataset. The project compares dropout rates from 0.1 to 0.5 while keeping the model architecture and training settings consistent.

## Project Overview

Dropout is a regularization technique that randomly disables a percentage of neuron outputs during training. This helps reduce overfitting by preventing the model from depending too heavily on individual neurons.

This notebook trains a neural network on MNIST using TensorFlow and Keras, then compares validation accuracy across multiple dropout rates. The model uses ReLU hidden layers, batch normalization, Adam optimization, categorical cross entropy loss, and a softmax output layer for digit classification.

## Dataset

The project uses the MNIST handwritten digit dataset.

```text
Training images: 60,000
Validation images: 10,000
Image size: 28 by 28 pixels
Flattened input size: 784 features
Number of classes: 10
```

## Preprocessing

The notebook prepares the data by:

- Reshaping each image from 28 by 28 pixels into a 784 value vector
- Converting image values to float32
- Normalizing pixel values from 0 to 255 into the range 0 to 1
- One hot encoding the digit labels into 10 output classes

## Model Architecture

The model uses the best hyperparameter setup from the previous class activity and adds dropout testing.

```text
Hidden layers: 3
Hidden units per layer: 64
Activation function: ReLU
Learning rate: 0.001
Optimizer: Adam
Epochs: 20
Batch size: 128
Regularization tested: Dropout
Additional layer: Batch Normalization
Output layer: Softmax with 10 classes
```

## Dropout Experiments

The notebook trains the same model architecture with five dropout rates.

| Hidden Layers | Activation Function | Learning Rate | Dropout Rate | Validation Accuracy |
| --- | --- | --- | --- | --- |
| 3 | ReLU | 0.001 | 0.1 | 97.43 percent |
| 3 | ReLU | 0.001 | 0.2 | 97.41 percent |
| 3 | ReLU | 0.001 | 0.3 | 97.33 percent |
| 3 | ReLU | 0.001 | 0.4 | 97.19 percent |
| 3 | ReLU | 0.001 | 0.5 | 97.03 percent |

## Best Result

The best validation result came from the lowest tested dropout rate.

```text
Best dropout: 0.1
Validation accuracy: 97.43 percent
```

## Key Findings

All tested dropout rates performed well and stayed above 97 percent validation accuracy.

The best result was dropout 0.1, which suggests that light regularization worked best for this model and dataset.

Higher dropout rates slightly reduced validation accuracy, likely because too much information was being removed during training.

The training history plots show that accuracy improved quickly during early epochs, while loss decreased as training progressed.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── Jason_Stys_CS478_01_CA7_Playing_with_drop_out.ipynb
├── Jason Stys CS478-01 CA7 Playing with drop out.pdf
├── README.md
└── requirements.txt
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/mnist-dropout-regularization-experiment.git
cd mnist-dropout-regularization-experiment
```

Install the required packages.

```bash
pip install tensorflow numpy pandas matplotlib notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

## Skills Demonstrated

- TensorFlow and Keras model development
- MNIST image classification
- Dropout regularization testing
- Batch normalization
- Neural network hyperparameter evaluation
- Training and validation accuracy comparison
- Accuracy and loss visualization
- pandas result table creation
- Jupyter Notebook workflow

## Possible Future Improvements

- Add a no dropout baseline for comparison
- Test dropout placement after every hidden layer
- Compare Adam with SGD
- Add early stopping
- Add confusion matrix evaluation
- Save the best trained model
- Add test set accuracy after final model selection
