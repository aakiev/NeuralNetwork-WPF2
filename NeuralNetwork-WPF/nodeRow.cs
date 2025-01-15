using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork_WPF
{
    class nodeRow
    {
        public string inputValue { get; set; }
        public string weightsIH { get; set; }
        public string inputHidden { get; set; }
        public string outputHidden { get; set; }
        public string weightsHO { get; set; }
        public string errorHidden { get; set; }
        public string inputOutput { get; set; }
        public string outputLayer { get; set; }
        public string target { get; set; }
        public string errorOutput { get; set; }
    }
}
