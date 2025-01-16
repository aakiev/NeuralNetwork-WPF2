# WPF Neural Network Image Recognizer

## Overview
This repository hosts a WPF-based application that implements a custom neural network for recognizing patterns in images. The application is built using C# and leverages the .NET framework for creating an intuitive graphical user interface. The project focuses on constructing a neural network from scratch and training it to identify digits or similar visual elements with high accuracy.

The neural network is designed to be modular and flexible, allowing for easy adjustments and improvements as the project evolves.

## Key Objectives
- Implement a fully functional neural network using C#
- Provide a clean and interactive WPF user interface
- Support training and testing on image datasets
- Ensure scalability for future enhancements, such as different types of input patterns or additional recognition tasks

## Core Features
1. **Neural Network Construction**  
   The project includes a customizable neural network architecture built from scratch in C#. Users can define key parameters such as the number of layers and neurons, activation functions, and training algorithms.
   
2. **Data Handling**  
   The application supports loading datasets for training and testing. The initial focus is on recognizing numerical patterns, but the structure is designed to accommodate various types of data.

3. **Real-Time Visualization**  
   The WPF interface offers real-time visualization of the neural network's performance during training and recognition, providing users with immediate feedback on accuracy and loss.

4. **Extensibility**  
   The architecture is designed to be modular, making it easy to introduce new features, modify the existing structure, or extend the recognition capabilities to different domains.

## Technologies Used
- **Programming Language**: C#
- **Framework**: .NET Framework (WPF)
- **Development Environment**: Visual Studio
- **Version Control**: Git & GitHub

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
Load a dataset and start the training process by configuring the neural network parameters in the UI.

### Recognition
Once trained, the network can be used to recognize patterns from new images.

### Visualization
The UI provides charts and metrics to monitor performance during training and testing.

## Future Enhancements
- Adding support for more complex datasets (e.g., handwritten letters or symbols)
- Implementing advanced neural network features like convolutional layers or recurrent architectures
- Enhancing the UI for better user interaction and customization
- Introducing export/import options for model weights
