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
        nn3S nn3SO;
        double[] inputs;

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

        private void createButton_Click(object sender, RoutedEventArgs e)
        {
            if ((inodes > 1) && (hnodes > 1) && (onodes > 1))
                nn3SO = new nn3S(inodes, hnodes, onodes);
        }

        private void queryButton_Click(object sender, RoutedEventArgs e)
        {
            int i, j, k;
            inputs = new double[inodes];

            inputs[0] = 0.9;
            inputs[1] = 0.1;
            inputs[2] = 0.8;

            nn3SO.queryNN(inputs);

            for (i = 0; i < inputs.Length; i++)
            {
                var data = new nodeRow
                {
                    inputValue = inputs[i].ToString(),
                    //weightsIH = "0",
                    inputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_inputs[i]),
                    outputHidden = String.Format(" {0:0.##} ", nn3SO.Hidden_outputs[i]),
                    //weightsHO = "0",
                    //errorHidden = "0",
                    inputOutput = String.Format(" {0:0.##} ", nn3SO.Final_inputs[i]),
                    outputLayer = String.Format(" {0:0.##} ", nn3SO.Final_outputs[i]),
                    //target = "0",
                    //errorOutput = "0",
                };
                
                networkDataGrid.Items.Add(data);
            }
        }
    }
}
