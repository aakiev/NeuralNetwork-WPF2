using System;
using System.IO;
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
using Microsoft.Win32;
using System.Drawing;

namespace NeuralNetwork_WPF
{
    public partial class MainWindow : Window
    {
        int inodes = 784, hnodes = 100, onodes = 10;
        int trainCount = 0, epoches = 1;
        double learningRate = 0.1;
        nn3S nn3SO;
        nnMath nnMathO = new nnMath();

        double[] inputs;
        double[] targets;

        string trainFile, testFile, weightFile;
        Boolean trainOK = false;

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
            if (inodes > 1 && hnodes > 1 && onodes > 1)
            {
                nn3SO = new nn3S(inodes, hnodes, onodes);
                MessageBox.Show($"Netzwerk erstellt: Eingänge: {inodes}, Hidden: {hnodes}, Ausgänge: {onodes}, Learningrate: {learningRate}");
            }
            else
            {
                MessageBox.Show("Die Anzahl der Neuronen muss größer als 1 sein!");
            }
        }

        private void trainButton_Click(object sender, RoutedEventArgs e)
        {
            int i, j;
            targets = new double[onodes];

            for (j = 0; j < epoches; j++)
                using (StreamReader sr = new StreamReader(trainFile))
                {
                    string line;
                    int intTarget;

                    while ((line = sr.ReadLine()) != null && (line != ""))
                    {
                        intTarget = readInputs(line);

                        for (i = 0; i < onodes; i++)
                        {
                            targets[i] = 0.01;
                        }

                        targets[intTarget] = 0.99;

                        trainCount++;
                        nn3SO.Train(inputs, targets, learningRate);
                        DisplayResults();

                        if (checkBoxImage.IsChecked == true) MessageBox.Show("Next");
                    }
                }

            weightFile = string.Concat("weight-", trainCount.ToString(), "-", epoches.ToString(), "-", hnodes.ToString());

            using (StreamWriter sw = new StreamWriter(weightFile + ".txt"))
            {
                sw.WriteLine($"wih {nn3SO.Wih.GetLength(0)} {nn3SO.Wih.GetLength(1)}");
                for (i = 0; i < nn3SO.Wih.GetLength(0); i++)
                {
                    for (j = 0; j < nn3SO.Wih.GetLength(1); j++)
                    {
                        sw.Write($"{nn3SO.Wih[i, j]} ");
                    }
                    sw.WriteLine();
                }

                sw.WriteLine($"who {nn3SO.Who.GetLength(0)} {nn3SO.Who.GetLength(1)}");
                for (i = 0; i < nn3SO.Who.GetLength(0); i++)
                {
                    for (j = 0; j < nn3SO.Who.GetLength(1); j++)
                    {
                        sw.Write($"{nn3SO.Who[i, j]} ");
                    }
                    sw.WriteLine();
                }
            }

            trainOK = true;
            MessageBox.Show("Training done: " + trainCount + " , with " + epoches + " epochs" );
        }

        private void openTrainButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                trainFile = openFileDialog.FileName;
                MessageBox.Show($"Trainingsdatei geladen: {trainFile}");
            }
        }

        public (double[,], double[,]) LoadWeights(string filePath)
        {
            using (StreamReader sr = new StreamReader(filePath))
            {
                string line = sr.ReadLine();
                string[] parts = line.Split();
                if (parts[0] != "wih")
                    throw new InvalidDataException("Expected 'wih' header.");
                int wihRows = int.Parse(parts[1]);
                int wihCols = int.Parse(parts[2]);

                double[,] wih = new double[wihRows, wihCols];
                for (int i = 0; i < wihRows; i++)
                {
                    line = sr.ReadLine();
                    parts = line.Split();
                    for (int j = 0; j < wihCols; j++)
                    {
                        wih[i, j] = double.Parse(parts[j]);
                    }
                }

                line = sr.ReadLine();
                parts = line.Split();
                if (parts[0] != "who")
                    throw new InvalidDataException("Expected 'who' header.");
                int whoRows = int.Parse(parts[1]);
                int whoCols = int.Parse(parts[2]);

                double[,] who = new double[whoRows, whoCols];
                for (int i = 0; i < whoRows; i++)
                {
                    line = sr.ReadLine();
                    parts = line.Split();
                    for (int j = 0; j < whoCols; j++)
                    {
                        who[i, j] = double.Parse(parts[j]);
                    }
                }

                return (wih, who);
            }
        }

        private void epochenBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            int.TryParse(epochenBox.Text, out int parsedEpoche);
            epoches = parsedEpoche;
        }

        private void epochenBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out int epoches);
        }

        private void loadWeightButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // Öffne den Dialog, um die Datei auszuwählen
                OpenFileDialog openFileDialog = new OpenFileDialog();
                openFileDialog.Filter = "Text files (*.txt)|*.txt|All files (*.*)|*.*";

                if (openFileDialog.ShowDialog() == true)
                {
                    // Datei laden
                    string weightFile = openFileDialog.FileName;

                    // Gewichte laden
                    (double[,] loadedWih, double[,] loadedWho) = LoadWeights(weightFile);

                    // Geladene Gewichte dem neuronalen Netz zuweisen
                    nn3SO.setWihMatrix(loadedWih);
                    nn3SO.setWhoMatrix(loadedWho);

                    MessageBox.Show("Gewichte erfolgreich geladen!", "Erfolg", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                else
                {
                    MessageBox.Show("Laden der Gewichte abgebrochen.", "Abbruch", MessageBoxButton.OK, MessageBoxImage.Warning);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Fehler beim Laden der Gewichte: {ex.Message}", "Fehler", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }


        private int readInputs(string line)
        {
            int i, j;
            inputs = new double[inodes];
            string[] input;
            byte[] inputsByte = new byte[inodes];

            input = line.Split(',');
            for (i = 1; i < input.Length; i++)
            {
                j = i - 1;
                inputs[j] = (Convert.ToDouble(input[i]) * 0.99 / 255.0) + 0.01;
                inputsByte[j] = (byte)Convert.ToInt32(input[i]);
            }
            if (checkBoxImage.IsChecked == true)
            {
                BitmapSource img = BitmapSource.Create(28, 28, 96, 96, PixelFormats.Indexed8, BitmapPalettes.Gray256, inputsByte, 28);
                numberImage.Source = img;
            }

            return Convert.ToInt32(input[0]);
        }

        private void openTestButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                testFile = openFileDialog.FileName;
                MessageBox.Show($"Test Datei geladen: {testFile}");
            }
        }

        private void queryButton_Click(object sender, RoutedEventArgs e)
        {
            int i;
            int scorecard = 0, testCounter = 0;
            targets = new double[onodes];
            using (StreamReader sr = new StreamReader(testFile))
            {
                string line;
                int intTarget, indexAnswer = 10;
                double answer = 0.0;
                while ((line = sr.ReadLine()) != null && (line != ""))
                {
                    answer = 0;
                    intTarget = readInputs(line);

                    for (i = 0; i < onodes; i++)
                    {
                        targets[i] = 0.01;
                    }

                    targets[intTarget] = 0.99;
                    nn3SO.queryNN(inputs);
                    foreach (var output in nn3SO.Final_outputs)
                    {
                        Console.WriteLine(output);
                    }


                    for (i = 0; i < nn3SO.Final_outputs.Length; i++)
                    {
                        if (nn3SO.Final_outputs[i] > answer)
                        {
                            answer = nn3SO.Final_outputs[i];
                            indexAnswer = i;
                        }
                    }

                    if (intTarget == indexAnswer)
                    {
                        scorecard++;
                    }

                    testCounter++;

                    DisplayResults();

                    if (checkBoxImage.IsChecked == true) MessageBox.Show("Next");
                }
            }

            performanceBox.Text = (scorecard / (double)testCounter).ToString();
        }
        private void DisplayResults()
        {
            int weightIHsize = nn3SO.Wih.Length / inodes; // Anzahl der Verbindungen pro Input-Neuron
            int weightHOsize = nn3SO.Who.Length / hnodes; // Anzahl der Verbindungen pro Hidden-Neuron

            networkDataGrid.Items.Clear();  // Daten-Grid zurücksetzen

            // Daten für networkDataGrid
            for (int i = 0; i < inodes; i++)
            {
                nodeRow data = new nodeRow();
                data.inputValue = inputs[i].ToString("F2"); // Eingabewert formatieren

                if (i < hnodes)
                {
                    data.inputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_inputs[i]);
                    data.outputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_outputs[i]);
                }
                if (i < onodes)
                {
                    data.inputOutput = String.Format(" {0:0.##} ", nn3SO.Final_inputs[i]);
                    data.outputLayer = String.Format(" {0:0.##} ", nn3SO.Final_outputs[i]);
                    data.target = targets[i].ToString("F2");
                }

                networkDataGrid.Items.Add(data);
            }

            // Ausgabe der erkannten Zahl
            double maxfound = nn3SO.Final_outputs[0];
            int indexMax = 0;
            for (int i = 1; i < onodes; i++)
            {
                if (nn3SO.Final_outputs[i] > maxfound)
                {
                    maxfound = nn3SO.Final_outputs[i];
                    indexMax = i;
                }
            }
            recognizedBox.Text = indexMax.ToString();
        }





    }
}
