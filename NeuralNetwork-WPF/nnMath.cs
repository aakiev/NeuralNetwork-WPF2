using System;

namespace NeuralNetwork_WPF
{
    public class nnMath
    {
        // Methods for matrix multiplication
        public double[,] matrixMult(double[,] matrixA, double[,] matrixB)
        {
            if (matrixA.GetLength(1) != matrixB.GetLength(0))
                throw new ArgumentException("Matrix A columns must match Matrix B rows.");

            int rows = matrixA.GetLength(0);
            int cols = matrixB.GetLength(1);
            int sharedDim = matrixA.GetLength(1);

            double[,] result = new double[rows, cols];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    for (int k = 0; k < sharedDim; k++)
                    {
                        result[i, j] += matrixA[i, k] * matrixB[k, j];
                    }
                }
            }
            return result;
        }

        public double[] matrixMult(double[,] matrix, double[] vector)
        {
            if (matrix.GetLength(1) != vector.Length)
                throw new ArgumentException("Matrix columns must match vector length.");

            int rows = matrix.GetLength(0);
            double[] result = new double[rows];

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < vector.Length; j++)
                {
                    result[i] += matrix[i, j] * vector[j];
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

        // Cost-Function als einfache Differenz
        public double[] CalculateOutputErrors(double[] targets, double[] outputs)
        {
            double[] errors = new double[targets.Length];
            
            for (int i = 0; i < targets.Length; i++)
            {
                errors[i] = targets[i] - outputs[i];
            }
            return errors;

        }

        // Fehler-Funktion für den Hidden Layer
        public double[] CalculateHiddenError(double[,] weights, double[] errorOutput)
        {
            if (weights == null)
                throw new ArgumentNullException(nameof(weights), "Die Gewichts-Matrix darf nicht null sein.");
            if (errorOutput == null)
                throw new ArgumentNullException(nameof(errorOutput), "Der Fehler-Array darf nicht null sein.");

            int rows = weights.GetLength(0); // Anzahl der Neuronen in der Hidden-Schicht
            int cols = weights.GetLength(1); // Anzahl der Neuronen in der Output-Schicht

            if (rows != errorOutput.Length)
                throw new ArgumentException("Die Anzahl der Spalten in der Gewichtsmatrix muss der Länge des Fehlervektors entsprechen.");

            double[] errorHidden = new double[cols];

            for (int i = 0; i < cols; i++)
            {
                errorHidden[i] = 0.0;
                for (int j = 0; j < errorOutput.Length; j++)
                {
                    errorHidden[i] += weights[j, i] * errorOutput[j];
                }
            }

            return errorHidden;
        }


    }
}
