using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_WPF
{
    public class nn4S // Neural Network 4 Layers
    {
        double[,] wih, whh, who; // Gewichtsmatrizen: Input-Hidden1, Hidden1-Hidden2, Hidden2-Output
        int inodes, hnodes1, hnodes2, onodes; // Anzahl der Neuronen in den Schichten
        double[] hidden1_inputs, hidden1_outputs;
        double[] hidden2_inputs, hidden2_outputs;
        double[] final_inputs, final_outputs;

        public double[] Hidden1_inputs { get { return hidden1_inputs; } }
        public double[] Hidden1_outputs { get { return hidden1_outputs; } }
        public double[] Hidden2_inputs { get { return hidden2_inputs; } }
        public double[] Hidden2_outputs { get { return hidden2_outputs; } }
        public double[] Final_inputs { get { return final_inputs; } }
        public double[] Final_outputs { get { return final_outputs; } }
        public double[,] Wih { get { return wih; } }
        public double[,] Whh { get { return whh; } }
        public double[,] Who { get { return who; } }

        public nn4S(int inodes, int hnodes1, int hnodes2, int onodes)
        {
            this.inodes = inodes;
            this.hnodes1 = hnodes1;
            this.hnodes2 = hnodes2;
            this.onodes = onodes;

            createWeightMatrizes();
        }

        private void createWeightMatrizes()
        {
            wih = new double[hnodes1, inodes];
            whh = new double[hnodes2, hnodes1];
            who = new double[onodes, hnodes2];

            Random random = new Random();

            // Initialisierung der Gewichte für Input -> Hidden1
            for (int j = 0; j < hnodes1; j++)
            {
                for (int i = 0; i < inodes; i++)
                {
                    wih[j, i] = random.NextDouble() * 2.0 - 1.0; // Werte im Bereich [-1.0, 1.0]
                }
            }

            // Initialisierung der Gewichte für Hidden1 -> Hidden2
            for (int j = 0; j < hnodes2; j++)
            {
                for (int i = 0; i < hnodes1; i++)
                {
                    whh[j, i] = random.NextDouble() * 2.0 - 1.0; // Werte im Bereich [-1.0, 1.0]
                }
            }

            // Initialisierung der Gewichte für Hidden2 -> Output
            for (int j = 0; j < onodes; j++)
            {
                for (int i = 0; i < hnodes2; i++)
                {
                    who[j, i] = random.NextDouble() * 2.0 - 1.0; // Werte im Bereich [-1.0, 1.0]
                }
            }
        }

        public void queryNN(double[] inputs)
        {
            nnMath nnMathO = new nnMath();

            // Input -> Hidden1
            hidden1_inputs = new double[hnodes1];
            hidden1_inputs = nnMathO.matrixMult(wih, inputs);

            hidden1_outputs = new double[hnodes1];
            hidden1_outputs = nnMathO.activationFunction(hidden1_inputs);

            // Hidden1 -> Hidden2
            hidden2_inputs = new double[hnodes2];
            hidden2_inputs = nnMathO.matrixMult(whh, hidden1_outputs);

            hidden2_outputs = new double[hnodes2];
            hidden2_outputs = nnMathO.activationFunction(hidden2_inputs);

            // Hidden2 -> Output
            final_inputs = new double[onodes];
            final_inputs = nnMathO.matrixMult(who, hidden2_outputs);

            final_outputs = new double[onodes];
            final_outputs = nnMathO.activationFunction(final_inputs);
        }

        public void Train(double[] inputs, double[] targets, double learningRate)
        {
            nnMath nnMathO = new nnMath();

            // Forward Pass
            queryNN(inputs);

            // Fehlerberechnung für Output
            double[] outputErrors = nnMathO.CalculateOutputErrors(targets, final_outputs);

            // Fehlerberechnung für Hidden2
            double[] hidden2Errors = nnMathO.CalculateHiddenError(who, outputErrors);

            // Fehlerberechnung für Hidden1
            double[] hidden1Errors = nnMathO.CalculateHiddenError(whh, hidden2Errors);

            // Gradienten für die Ausgabeschicht
            double[] outputGradients = new double[onodes];
            for (int i = 0; i < onodes; i++)
            {
                outputGradients[i] = outputErrors[i] * final_outputs[i] * (1 - final_outputs[i]);
            }

            // Gradienten für Hidden2
            double[] hidden2Gradients = new double[hnodes2];
            for (int i = 0; i < hnodes2; i++)
            {
                hidden2Gradients[i] = hidden2Errors[i] * hidden2_outputs[i] * (1 - hidden2_outputs[i]);
            }

            // Gradienten für Hidden1
            double[] hidden1Gradients = new double[hnodes1];
            for (int i = 0; i < hnodes1; i++)
            {
                hidden1Gradients[i] = hidden1Errors[i] * hidden1_outputs[i] * (1 - hidden1_outputs[i]);
            }

            // Gewichtsanpassung für Hidden2 -> Output
            for (int i = 0; i < hnodes2; i++)
            {
                for (int j = 0; j < onodes; j++)
                {
                    who[j, i] += learningRate * outputGradients[j] * hidden2_outputs[i];
                }
            }

            // Gewichtsanpassung für Hidden1 -> Hidden2
            for (int i = 0; i < hnodes1; i++)
            {
                for (int j = 0; j < hnodes2; j++)
                {
                    whh[j, i] += learningRate * hidden2Gradients[j] * hidden1_outputs[i];
                }
            }

            // Gewichtsanpassung für Input -> Hidden1
            for (int i = 0; i < inodes; i++)
            {
                for (int j = 0; j < hnodes1; j++)
                {
                    wih[j, i] += learningRate * hidden1Gradients[j] * inputs[i];
                }
            }
        }

        public void setWihMatrix(double[,] wih)
        {
            this.wih = wih;
        }

        public void setWhhMatrix(double[,] whh)
        {
            this.whh = whh;
        }

        public void setWhoMatrix(double[,] who)
        {
            this.who = who;
        }

    }
}
