
# WPF Neural Network Image Recognizer

## Overview
This repository hosts a WPF-based application that implements a custom neural network for recognizing patterns in images. The application is built using C# and leverages the .NET framework for creating an intuitive graphical user interface. The project focuses on constructing a neural network from scratch and training it to identify digits or clothing items with high accuracy using datasets such as MNIST and Zalando's Fashion-MNIST.

The neural network is designed to be modular and flexible, allowing for easy adjustments and improvements as the project evolves.

## Key Objectives
- Implement a fully functional neural network using C#
- Provide a clean and interactive WPF user interface
- Support training and testing on image datasets (MNIST digits and Zalando Fashion-MNIST)
- Ensure scalability for future enhancements, such as different types of input patterns or additional recognition tasks

## Core Features
1. **Neural Network Construction**  
   The project includes a customizable neural network architecture built from scratch in C#. Users can define key parameters such as the number of layers and neurons, activation functions, and training algorithms. The network can handle configurations with 1, 2, or 3 hidden layers, with the flexibility to experiment with different architectures.

2. **Data Handling**  
   The application supports loading datasets for training and testing. It currently focuses on the MNIST digit dataset and Zalando’s Fashion-MNIST dataset for clothing image recognition. The structure is designed to easily accommodate different types of data in the future.

3. **Real-Time Visualization**  
   The WPF interface offers real-time visualization of the neural network's performance during training and recognition. Users receive immediate feedback on accuracy and loss, as well as the network's performance through the user interface, including metrics and visualized images for training samples.

4. **Extensibility**  
   The architecture is designed to be modular, making it easy to introduce new features, modify the existing structure, or extend the recognition capabilities to different domains (e.g., handwritten letters or other object recognition tasks).

## Technologies Used
- **Programming Language**: C#
- **Framework**: .NET Framework (WPF)
- **Development Environment**: Visual Studio 2019 or later
- **Version Control**: Git & GitHub
- **Libraries**: Matplotlib (for performance plotting), Hyperref (for hyperlinks in LaTeX)

## Getting Started

### Prerequisites
- Visual Studio 2019 or later
- .NET Framework installed on your machine

### Installation
1. Clone the repository using the following command:
   
   ```bash
   git clone https://github.com/aakiev/NeuralNetwork-WPF.git
   ```
2. Open the solution file (.sln) in Visual Studio.
   
3. Build the solution and run the application.

## Usage

### Training
Load a dataset (MNIST or Fashion-MNIST) and start the training process by configuring the neural network parameters in the UI. The application allows you to select the number of hidden layers, the learning rate, and the number of epochs for training.

### Recognition
Once trained, the network can be used to recognize patterns from new images, either from the same dataset or new input images.

### Visualization
The UI provides charts and metrics to monitor performance during training and testing. Users can visualize the performance of the network and track its accuracy over time.

### Performance Comparison
The project includes the ability to experiment with different network architectures. Performance is tested using the MNIST digit dataset and Zalando Fashion-MNIST dataset, providing insights into how increasing the number of hidden layers affects accuracy. Example results include:
- 1 hidden layer: ~93.85% on MNIST, ~81.55% on Fashion-MNIST
- 2 hidden layers: ~88.38% on MNIST, ~65.45% on Fashion-MNIST
- 3 hidden layers: ~54.49% on MNIST, ~19.81% on Fashion-MNIST

## Future Enhancements
- Adding support for more complex datasets (e.g., handwritten letters or symbols)
- Implementing advanced neural network features like convolutional layers or recurrent architectures
- Enhancing the UI for better user interaction and customization
- Introducing export/import options for model weights to save and load trained networks
- Increasing training epochs and experimenting with different learning rates for improved performance
