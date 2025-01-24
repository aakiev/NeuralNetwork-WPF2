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
        int hiddenLayerCount;

        nn3S nn3SO;
        nn4S nn4SO;
        nn5S nn5SO;
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
                if (hiddenLayerCount == 1)
                {
                    nn3SO = new nn3S(inodes, hnodes, onodes);
                    MessageBox.Show($"3-Schicht-Netzwerk erstellt: Eingänge: {inodes} | Hidden: {hnodes} | Ausgänge: {onodes} | Lernrate: {learningRate} | Epochen: {epoches}");
                }
                else if (hiddenLayerCount == 2)
                {
                    nn4SO = new nn4S(inodes, hnodes, hnodes, onodes);
                    MessageBox.Show($"4-Schicht-Netzwerk erstellt: Eingänge: {inodes} | Hidden 1: {hnodes} | Hidden 2: {hnodes} | Ausgänge: {onodes} | Lernrate: {learningRate} | Epochen: {epoches}");
                }
                else if (hiddenLayerCount == 3)
                {
                    nn5SO = new nn5S(inodes, hnodes, hnodes, hnodes, onodes);
                    MessageBox.Show($"5-Schicht-Netzwerk erstellt: Eingänge: {inodes} | Hidden 1: {hnodes} | Hidden 2: {hnodes} | Hidden 3: {hnodes} | Ausgänge: {onodes} | Lernrate: {learningRate} | Epochen: {epoches}");
                }
                else
                {
                    MessageBox.Show("Anzahl der Hidden-Layer wird nicht unterstützt!");
                    return;
                }

                openTrainButton.IsEnabled = true;
                loadWeightButton.IsEnabled = true;
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
            {
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

                        if (hiddenLayerCount == 1)
                        {
                            nn3SO.Train(inputs, targets, learningRate);
                        }
                        else if (hiddenLayerCount == 2)
                        {
                            nn4SO.Train(inputs, targets, learningRate);
                        }
                        else if (hiddenLayerCount == 3)
                        {
                            nn5SO.Train(inputs, targets, learningRate);
                        }
                        else
                        {
                            MessageBox.Show("Anzahl der Hidden-Layer wird nicht unterstützt!");
                            return;
                        }

                        DisplayResults();

                        if (checkBoxImage.IsChecked == true) MessageBox.Show("Next");
                    }
                }
            }

            weightFile = string.Concat("weight-", trainCount.ToString(), "-", epoches.ToString(), "-", hnodes.ToString(), "-", hiddenLayerCount.ToString());

            using (StreamWriter sw = new StreamWriter(weightFile + ".txt"))
            {
                if (hiddenLayerCount == 1)
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
                else if (hiddenLayerCount == 2)
                {
                    sw.WriteLine($"wih {nn4SO.Wih.GetLength(0)} {nn4SO.Wih.GetLength(1)}");
                    for (i = 0; i < nn4SO.Wih.GetLength(0); i++)
                    {
                        for (j = 0; j < nn4SO.Wih.GetLength(1); j++)
                        {
                            sw.Write($"{nn4SO.Wih[i, j]} ");
                        }
                        sw.WriteLine();
                    }

                    sw.WriteLine($"whh {nn4SO.Whh.GetLength(0)} {nn4SO.Whh.GetLength(1)}");
                    for (i = 0; i < nn4SO.Whh.GetLength(0); i++)
                    {
                        for (j = 0; j < nn4SO.Whh.GetLength(1); j++)
                        {
                            sw.Write($"{nn4SO.Whh[i, j]} ");
                        }
                        sw.WriteLine();
                    }

                    sw.WriteLine($"who {nn4SO.Who.GetLength(0)} {nn4SO.Who.GetLength(1)}");
                    for (i = 0; i < nn4SO.Who.GetLength(0); i++)
                    {
                        for (j = 0; j < nn4SO.Who.GetLength(1); j++)
                        {
                            sw.Write($"{nn4SO.Who[i, j]} ");
                        }
                        sw.WriteLine();
                    }
                }
                else if (hiddenLayerCount == 3)
                {
                    sw.WriteLine($"wih {nn5SO.Wih.GetLength(0)} {nn5SO.Wih.GetLength(1)}");
                    for (i = 0; i < nn5SO.Wih.GetLength(0); i++)
                    {
                        for (j = 0; j < nn5SO.Wih.GetLength(1); j++)
                        {
                            sw.Write($"{nn5SO.Wih[i, j]} ");
                        }
                        sw.WriteLine();
                    }

                    sw.WriteLine($"whh1 {nn5SO.Whh1.GetLength(0)} {nn5SO.Whh1.GetLength(1)}");
                    for (i = 0; i < nn5SO.Whh1.GetLength(0); i++)
                    {
                        for (j = 0; j < nn5SO.Whh1.GetLength(1); j++)
                        {
                            sw.Write($"{nn5SO.Whh1[i, j]} ");
                        }
                        sw.WriteLine();
                    }

                    sw.WriteLine($"whh2 {nn5SO.Whh2.GetLength(0)} {nn5SO.Whh2.GetLength(1)}");
                    for (i = 0; i < nn5SO.Whh2.GetLength(0); i++)
                    {
                        for (j = 0; j < nn5SO.Whh2.GetLength(1); j++)
                        {
                            sw.Write($"{nn5SO.Whh2[i, j]} ");
                        }
                        sw.WriteLine();
                    }

                    sw.WriteLine($"who {nn5SO.Who.GetLength(0)} {nn5SO.Who.GetLength(1)}");
                    for (i = 0; i < nn5SO.Who.GetLength(0); i++)
                    {
                        for (j = 0; j < nn5SO.Who.GetLength(1); j++)
                        {
                            sw.Write($"{nn5SO.Who[i, j]} ");
                        }
                        sw.WriteLine();
                    }
                }
            }

            trainOK = true;
            openTestButton.IsEnabled = true;
            MessageBox.Show("Training abgeschlossen: " + trainCount + " Trainingsdurchläufe, " + epoches + " Epochen");
        }


        private void openTrainButton_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            if (openFileDialog.ShowDialog() == true)
            {
                trainFile = openFileDialog.FileName;
                MessageBox.Show($"Trainingsdatei geladen: {trainFile}");
            }

            if (trainFile != "") trainButton.IsEnabled = true;
        }

        public (double[,], double[,], double[,], double[,]) LoadWeights(string filePath)
        {
            using (StreamReader sr = new StreamReader(filePath))
            {
                double[,] wih = null, whh = null, who = null, whh2 = null;

                // Lade wih
                string line = sr.ReadLine();
                string[] parts = line.Split();
                if (parts[0] != "wih")
                    throw new InvalidDataException("Expected 'wih' header.");
                int wihRows = int.Parse(parts[1]);
                int wihCols = int.Parse(parts[2]);

                wih = new double[wihRows, wihCols];
                for (int i = 0; i < wihRows; i++)
                {
                    line = sr.ReadLine();
                    parts = line.Split();
                    for (int j = 0; j < wihCols; j++)
                    {
                        wih[i, j] = double.Parse(parts[j]);
                    }
                }

                // Lade optional whh
                line = sr.ReadLine();
                if (line != null && line.StartsWith("whh"))
                {
                    parts = line.Split();
                    int whhRows = int.Parse(parts[1]);
                    int whhCols = int.Parse(parts[2]);

                    whh = new double[whhRows, whhCols];
                    for (int i = 0; i < whhRows; i++)
                    {
                        line = sr.ReadLine();
                        parts = line.Split();
                        for (int j = 0; j < whhCols; j++)
                        {
                            whh[i, j] = double.Parse(parts[j]);
                        }
                    }

                    // Prüfe ob whh2 folgt
                    line = sr.ReadLine();
                    if (line != null && line.StartsWith("whh2"))
                    {
                        parts = line.Split();
                        int whh2Rows = int.Parse(parts[1]);
                        int whh2Cols = int.Parse(parts[2]);

                        whh2 = new double[whh2Rows, whh2Cols];
                        for (int i = 0; i < whh2Rows; i++)
                        {
                            line = sr.ReadLine();
                            parts = line.Split();
                            for (int j = 0; j < whh2Cols; j++)
                            {
                                whh2[i, j] = double.Parse(parts[j]);
                            }
                        }

                        // Lade who
                        line = sr.ReadLine();
                    }
                }

                if (line == null || !line.StartsWith("who"))
                    throw new InvalidDataException("Expected 'who' header.");
                parts = line.Split();
                int whoRows = int.Parse(parts[1]);
                int whoCols = int.Parse(parts[2]);

                who = new double[whoRows, whoCols];
                for (int i = 0; i < whoRows; i++)
                {
                    line = sr.ReadLine();
                    parts = line.Split();
                    for (int j = 0; j < whoCols; j++)
                    {
                        who[i, j] = double.Parse(parts[j]);
                    }
                }

                return (wih, whh, whh2, who);
            }
        }

        private void epochenBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            int.TryParse(epochenBox.Text, out int parsedEpoche);
            epoches = parsedEpoche;
        }

        private void epochenBox_PreviewTextInput(object sender, TextCompositionEventArgs e)
        {
            e.Handled = !int.TryParse(e.Text, out _);
        }

        private void epochenBox_GotFocus(object sender, RoutedEventArgs e)
        {
            TextBox textBox = sender as TextBox;
            textBox.Text = "";
        }

        private void epochenBox_LostFocus(object sender, RoutedEventArgs e)
        {
            TextBox textBox = sender as TextBox;
            if (textBox.Text == "")
            {
                textBox.Text = "1";
            }
        }

        private void hiddenTextBox_LostFocus(object sender, RoutedEventArgs e)
        {
            TextBox textBox = sender as TextBox;
            if (textBox.Text == "")
            {
                textBox.Text = "100";
            }
        }

        private void hiddenTextBox_GotFocus(object sender, RoutedEventArgs e)
        {
            TextBox textBox = sender as TextBox;
            textBox.Text = "";
        }

        private void hiddenTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            int.TryParse(hiddenTextBox.Text, out int parsedHiddenNodes);
            hnodes = parsedHiddenNodes;
        }

        private void ComboBoxHiddenLayers_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            hiddenLayerCount = ComboBoxHiddenLayers.SelectedIndex + 1;
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
                    var weights = LoadWeights(openFileDialog.FileName);

                    if (hiddenLayerCount == 1)
                    {
                        nn3SO.setWihMatrix(weights.Item1);
                        nn3SO.setWhoMatrix(weights.Item4);
                    }
                    else if (hiddenLayerCount == 2)
                    {
                        nn4SO.setWihMatrix(weights.Item1);
                        nn4SO.setWhhMatrix(weights.Item2);
                        nn4SO.setWhoMatrix(weights.Item4);
                    }
                    else if (hiddenLayerCount == 3)
                    {
                        nn5SO.setWihMatrix(weights.Item1);
                        nn5SO.setWhh1Matrix(weights.Item2);
                        nn5SO.setWhh2Matrix(weights.Item3);
                        nn5SO.setWhoMatrix(weights.Item4);
                    }

                    openTestButton.IsEnabled = true;
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

            if (testFile != "") queryButton.IsEnabled = true;
        }

        private void queryButton_Click(object sender, RoutedEventArgs e)
        {
            int i;
            int scorecard = 0, testCounter = 0;
            targets = new double[onodes];

            using (StreamReader sr = new StreamReader(testFile))
            {
                string line;
                int intTarget, indexAnswer = -1;
                double answer = 0.0;

                while ((line = sr.ReadLine()) != null && (line != ""))
                {
                    answer = 0.0;
                    intTarget = readInputs(line);

                    // Zielvektor initialisieren
                    for (i = 0; i < onodes; i++)
                    {
                        targets[i] = 0.01;
                    }
                    targets[intTarget] = 0.99;

                    double[] finalOutputs;

                    // Abfrage des entsprechenden Netzwerks
                    if (hiddenLayerCount == 1)
                    {
                        nn3SO.queryNN(inputs);
                        finalOutputs = nn3SO.Final_outputs;
                    }
                    else if (hiddenLayerCount == 2 && nn4SO != null)
                    {
                        nn4SO.queryNN(inputs);
                        finalOutputs = nn4SO.Final_outputs;
                    }
                    else if (hiddenLayerCount == 3 && nn5SO != null)
                    {
                        nn5SO.queryNN(inputs);
                        finalOutputs = nn5SO.Final_outputs;
                    }
                    else
                    {
                        MessageBox.Show("Anzahl der Hidden-Layer wird nicht unterstützt oder Netzwerk wurde nicht erstellt!");
                        return;
                    }

                    // Ausgabe der Ergebnisse für Debugging
                    foreach (var output in finalOutputs)
                    {
                        Console.WriteLine(output);
                    }

                    // Erkennung der Zahl basierend auf den höchsten Werten
                    for (i = 0; i < finalOutputs.Length; i++)
                    {
                        if (finalOutputs[i] > answer)
                        {
                            answer = finalOutputs[i];
                            indexAnswer = i;
                        }
                    }

                    // Prüfen, ob die erkannte Zahl korrekt ist
                    if (intTarget == indexAnswer)
                    {
                        scorecard++;
                    }

                    testCounter++;

                    // Ergebnisse anzeigen
                    DisplayResults();

                    // Optional: Nachricht für den nächsten Schritt
                    if (checkBoxImage.IsChecked == true)
                    {
                        MessageBox.Show("Next");
                    }
                }
            }

            // Leistung des Netzwerks anzeigen
            performanceBox.Text = (scorecard / (double)testCounter).ToString();
        }

        private void DisplayResults()
        {
            networkDataGrid.Items.Clear(); // Daten-Grid zurücksetzen

            if (hiddenLayerCount == 1)
            {
                // Für nn3S
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
            }
            else if (hiddenLayerCount == 2)
            {
                // Für nn4S
                for (int i = 0; i < inodes; i++)
                {
                    nodeRow data = new nodeRow();
                    data.inputValue = inputs[i].ToString("F2"); // Eingabewert formatieren

                    if (i < hnodes)
                    {
                        data.inputHidden = String.Format(" {0:0.##} ", nn4SO.Hidden1_inputs[i]);
                        data.outputHidden = String.Format(" {0:0.##} ", nn4SO.Hidden2_outputs[i]);
                    }

                    if (i < onodes)
                    {
                        data.inputOutput = String.Format(" {0:0.##} ", nn4SO.Final_inputs[i]);
                        data.outputLayer = String.Format(" {0:0.##} ", nn4SO.Final_outputs[i]);
                        data.target = targets[i].ToString("F2");
                    }

                    networkDataGrid.Items.Add(data);
                }
            }
            else if (hiddenLayerCount == 3)
            {
                // Für nn5S
                for (int i = 0; i < inodes; i++)
                {
                    nodeRow data = new nodeRow();
                    data.inputValue = inputs[i].ToString("F2"); // Eingabewert formatieren

                    if (i < hnodes)
                    {
                        data.inputHidden = String.Format(" {0:0.##} ", nn5SO.Hidden1_inputs[i]);
                        data.outputHidden = String.Format(" {0:0.##} ", nn5SO.Hidden3_outputs[i]);
                    }

                    if (i < onodes)
                    {
                        data.inputOutput = String.Format(" {0:0.##} ", nn5SO.Final_inputs[i]);
                        data.outputLayer = String.Format(" {0:0.##} ", nn5SO.Final_outputs[i]);
                        data.target = targets[i].ToString("F2");
                    }

                    networkDataGrid.Items.Add(data);
                }
            }

            // Ausgabe der erkannten Zahl
            double maxfound = 0.0;
            int indexMax = 0;

            if (hiddenLayerCount == 1)
            {
                maxfound = nn3SO.Final_outputs[0];
                for (int i = 1; i < onodes; i++)
                {
                    if (nn3SO.Final_outputs[i] > maxfound)
                    {
                        maxfound = nn3SO.Final_outputs[i];
                        indexMax = i;
                    }
                }
            }
            else if (hiddenLayerCount == 2)
            {
                maxfound = nn4SO.Final_outputs[0];
                for (int i = 1; i < onodes; i++)
                {
                    if (nn4SO.Final_outputs[i] > maxfound)
                    {
                        maxfound = nn4SO.Final_outputs[i];
                        indexMax = i;
                    }
                }
            }
            else if (hiddenLayerCount == 3)
            {
                maxfound = nn5SO.Final_outputs[0];
                for (int i = 1; i < onodes; i++)
                {
                    if (nn5SO.Final_outputs[i] > maxfound)
                    {
                        maxfound = nn5SO.Final_outputs[i];
                        indexMax = i;
                    }
                }
            }

            recognizedBox.Text = indexMax.ToString();
        }
    }
}
