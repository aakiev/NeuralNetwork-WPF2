using System;

namespace NeuralNetwork_WPF
{
    public class nnMath
    {
        // Method for matrix multiplication
        public double[] matrixMult(double[,] matrix, int columns, double[] vector)
        {
            int rows = matrix.GetLength(0); // Number of rows in the matrix
            double[] result = new double[rows];

            for (int i = 0; i < rows; i++)
            {
                result[i] = 0;
                for (int j = 0; j < columns; j++)
                {
                    result[i] += matrix[j, i] * vector[j];
                }
            }

            return result;
        }

        // Sigmoid activation function
        public double[] activationFunction(double[] inputs)
        {
            double[] outputs = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                outputs[i] = 1.0 / (1.0 + Math.Exp(-inputs[i]));
            }
            return outputs;
        }
    }
}
