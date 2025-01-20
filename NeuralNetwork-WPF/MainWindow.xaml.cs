using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuralNetwork_WPF
{
    public partial class MainWindow : Window
    {
        int inodes = 3, hnodes = 3, onodes = 3;
        double learningRate = 0.1;
        nn3S nn3SO;
        nnMath nnMathO = new nnMath();

        double[] inputs;
        double[] targets;
        double[] errorsOutput;
        double[] errorsHidden;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void inputTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int inodes);
        }

        private void hiddenTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int hnodes);
        }

        private void outputTextBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int onodes);
        }

        private void TextBox_LearningRate_TextChanged(object sender, TextChangedEventArgs e)
        {
            if (double.TryParse(((TextBox)sender).Text, System.Globalization.NumberStyles.AllowDecimalPoint, System.Globalization.CultureInfo.InvariantCulture, out double newLearningRate))
            {
                learningRate = newLearningRate;
            }
        }


        private void TextBox_LearningRate_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            // Erlaubt nur Zahlen und maximal einen Dezimalpunkt
            e.Handled = !IsValidLearningRateInput(e.Text, ((TextBox)sender).Text);
        }

        // Validierungsmethode
        private bool IsValidLearningRateInput(string newText, string currentText)
        {
            // Erlaubt nur Zahlen und einen Dezimalpunkt
            string combinedText = currentText + newText;
            return double.TryParse(combinedText, System.Globalization.NumberStyles.AllowDecimalPoint, System.Globalization.CultureInfo.InvariantCulture, out _);
        }

        private void TextBox_LearningRate_LostFocus(object sender, RoutedEventArgs e)
        {
            TextBox textBox = sender as TextBox;
            if (textBox.Text == "") // Überprüft, ob das Textfeld leer ist
            {
                textBox.Text = "0.1"; // Platzhaltertext setzen
            }
        }

        private void TextBox_LearningRate_GotFocus(object sender, RoutedEventArgs e)
        {
            TextBox textBox = sender as TextBox;
            textBox.Text = ""; 
        }
        private void createButton_Click(object sender, RoutedEventArgs e)
        {
            if ((inodes > 1) && (hnodes > 1) && (onodes > 1))
                nn3SO = new nn3S(inodes, hnodes, onodes);
        }
        private void trainButton_Click(object sender, RoutedEventArgs e)
        {
            if (nn3SO == null)
            {
                MessageBox.Show("Bitte erstellen Sie zuerst ein neuronales Netz!", "Fehler", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            inputs = new double[inodes];
            targets = new double[onodes];

            inputs[0] = 0.9;
            inputs[1] = 0.1;
            inputs[2] = 0.8;

            targets[0] = 0.9; ;
            targets[1] = 0.9;
            targets[2] = 0.9;

            // Training durchführen
            nn3SO.Train(inputs, targets, learningRate);

            // Netzwerk erneut abfragen, um die aktualisierten Werte zu sehen
            nn3SO.queryNN(inputs);
            errorsOutput = nnMathO.CalculateOutputErrors(targets, nn3SO.Final_outputs);
            errorsHidden = nnMathO.CalculateHiddenError(nn3SO.Wih, errorsOutput);

            DisplayResults();
        }

        private void queryButton_Click(object sender, RoutedEventArgs e)
        {

            if (nn3SO == null)
            {
                MessageBox.Show("Bitte erstellen Sie zuerst ein neuronales Netz!", "Fehler", MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            inputs = new double[inodes];
            targets = new double[onodes];

            inputs[0] = 0.9;
            inputs[1] = 0.1;
            inputs[2] = 0.8;

            targets[0] = 0.9;
            targets[1] = 0.9;
            targets[2] = 0.9;

            nn3SO.queryNN(inputs);
            errorsOutput = nnMathO.CalculateOutputErrors(targets, nn3SO.Final_outputs);
            errorsHidden = nnMathO.CalculateHiddenError(nn3SO.Wih, errorsOutput);

            DisplayResults();
        }
        private void DisplayResults()
        {
            // Ergebnisse anzeigen
            networkDataGrid.Items.Clear();
            networkDataGrid_2.Items.Clear();

            for (int i = 0; i < inputs.Length; i++)
            {
                // Gewichtsinformationen für die aktuelle Zeile 
                string weightsIHForNeuron = String.Join(" | ", Enumerable.Range(0, nn3SO.Wih.GetLength(1))
                    .Select(j => nn3SO.Wih[i, j].ToString("F2")));

                string weightsHOForNeuron = String.Join(" | ", Enumerable.Range(0, nn3SO.Who.GetLength(1))
                    .Select(j => nn3SO.Who[i, j].ToString("F2")));

                var data = new nodeRow
                {
                    inputValue = inputs[i].ToString("F2"),
                    weightsIH = weightsIHForNeuron, 
                    inputHidden = nn3SO.Hidden_inputs[i].ToString("F2"),
                    outputHidden = nn3SO.Hidden_outputs[i].ToString("F2"),
                    weightsHO = weightsHOForNeuron, 
                    errorHidden = errorsHidden[i].ToString("F3"),
                    inputOutput = nn3SO.Final_inputs[i].ToString("F2"),
                    outputLayer = nn3SO.Final_outputs[i].ToString("F2"),
                    target = targets[i].ToString("F2"),
                    errorOutput = errorsOutput[i].ToString("F3"),
                };

                networkDataGrid.Items.Add(data);
                networkDataGrid_2.Items.Add(data);
            }
        }



    }
}
