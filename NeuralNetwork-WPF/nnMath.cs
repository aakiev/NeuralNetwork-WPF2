﻿using System;

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

        // Cost-Function als quadratischen Fehler mit Vorzeichen
        public double[] CalculateOutputErrors(double[] targets, double[] outputs)
        {
            double[] test = new double[targets.Length];
            double[] errors = new double[targets.Length];
            int a;
            
            for (int i = 0; i < targets.Length; i++)
            {
                test[i] = targets[i] - outputs[i];

                if (test[i] < 0)
                {
                    a = -1; 
                }
                else
                {
                    a = 1; 

                }

                errors[i] = a*Math.Pow(targets[i] - outputs[i],2);
            }
            return errors;

        }

        // Fehler-Funktion für den Hidden Layer
        public double[] CalculateHiddenError(double[,] weights, double[] errorOutput)
        {
            int rows = weights.GetLength(0); // Anzahl der Neuronen in der Hidden-Schicht
            int cols = weights.GetLength(1); // Anzahl der Neuronen in der Output-Schicht

            if (cols != errorOutput.Length)
            {
                throw new ArgumentException("Die Anzahl der Spalten der Gewichtsmatrix muss der Länge des Fehlervektors entsprechen.");
            }

            double[] errorHidden = new double[rows];

            for (int i = 0; i < rows; i++)
            {
                errorHidden[i] = 0.0;
                for (int j = 0; j < cols; j++)
                {
                    errorHidden[i] += weights[i, j] * errorOutput[j];
                }
            }

            return errorHidden;

        }
    }
}
