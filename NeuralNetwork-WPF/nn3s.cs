using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_WPF
{
    class nn3S
    {
        double[,] wih, who;
        int inodes, hnodes, onodes;
        double[] hidden_inputs;
        double[] hidden_outputs;
        double[] final_inputs;
        double[] final_outputs;

        public double[] Hidden_inputs { get { return hidden_inputs; } }
        public double[] Hidden_outputs { get { return hidden_outputs; } }
        public double[] Final_inputs { get { return final_inputs; } }
        public double[] Final_outputs { get { return final_outputs; } }

        public nn3S(int inodes, int hnodes, int onodes)
        {
            this.inodes = inodes;
            this.hnodes = hnodes;
            this.onodes = onodes;

            createWeightMatrizes();
        }

        private void createWeightMatrizes()
        {
            wih = new double[inodes, hnodes];
            who = new double[hnodes, onodes];

            wih[0, 0] = 0.9;    //Gewicht von Neuron 1 vom Inputlayer zu Neuron 1 vom Hiddenlayer
            wih[1, 0] = 0.3;    //Gewicht von Neuron 2 vom Inputlayer zu Neuron 1 vom Hiddenlayer
            wih[2, 0] = 0.4;    //Gewicht von Neuron 3 vom Inputlayer zu Neuron 1 vom Hiddenlayer
            wih[0, 1] = 0.2;
            wih[1, 1] = 0.8;
            wih[2, 1] = 0.2;
            wih[0, 2] = 0.1;
            wih[1, 2] = 0.5;
            wih[2, 2] = 0.6;

            who[0, 0] = 0.3;
            who[1, 0] = 0.7;
            who[2, 0] = 0.5;
            who[0, 1] = 0.6;
            who[1, 1] = 0.5;
            who[2, 1] = 0.2;
            who[0, 2] = 0.8;
            who[1, 2] = 0.1;
            who[2, 2] = 0.9;

            /*for (int j = 0; j < hnodes; j++)
                for (int i = 0; i < inodes; i++)
                {
                    System.Random weight_ih = new System.Random();
                    wih[i, j] = weight_ih.NextDouble() - 0.5;
                }
            for (int j = 0; j < onodes; j++)
                for (int i = 0; i < hnodes; i++)
                {
                    System.Random weight_ho = new System.Random();
                    who[i, j] = weight_ho.NextDouble() - 0.5;
                }*/
        }

        public void queryNN(double[] inputs)
        {
            nnMath nnMathO = new nnMath();

            hidden_inputs = new double[hnodes];
            hidden_inputs = nnMathO.matrixMult(wih, inodes, inputs);

            hidden_outputs = new double[hnodes];
            hidden_outputs = nnMathO.activationFunction(hidden_inputs);

            final_inputs = new double[onodes];
            final_inputs = nnMathO.matrixMult(who, hnodes, hidden_outputs);

            final_outputs = new double[hnodes];
            final_outputs = nnMathO.activationFunction(final_inputs);
        }
    }
}

