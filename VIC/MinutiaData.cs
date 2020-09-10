using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace VIC
{

    /*Model for the minutiae data, some consecutives columns are loaded as a vercto instead of individually*/

    public class MinutiaData
    {
        [LoadColumn(0)]
        public string fingerprint { get; set; }

        [LoadColumn(1)]
        public string minutia { get; set; }

        [LoadColumn(2, 42)]
        [VectorType(41)]
        public float[] nn { get; set; }

        [LoadColumn(43, 83)]
        [VectorType(41)]
        public float[] nnr { get; set; }

        [LoadColumn(84, 123)]
        [VectorType(40)]
        public float[] nn_nn { get; set; }

        [LoadColumn(124, 163)]
        [VectorType(40)]
        public float[] nnr_nnr { get; set; }

        [LoadColumn(164, 175)]
        [VectorType(12)]
        public float[] dn { get; set; }

        [LoadColumn(176)]
        public float df { get; set; }

        [LoadColumn(177, 188)]
        [VectorType(12)]
        public float[] dnr { get; set; }

        [LoadColumn(189)]
        public float dfr { get; set; }

        [LoadColumn(190, 201)]
        [VectorType(12)]
        public float[] dn_dn { get; set; }

        [LoadColumn(202, 213)]
        [VectorType(12)]
        public float[] dnr_dnr { get; set; }

        [LoadColumn(214, 225)]
        [VectorType(12)]
        public float[] alphan { get; set; }

        [LoadColumn(226)]
        public float alphaf { get; set; }

        [LoadColumn(227, 238)]
        [VectorType(12)]
        public float[] alphann { get; set; }

        [LoadColumn(239)]
        public float alphanf { get; set; }

        [LoadColumn(240, 251)]
        [VectorType(12)]
        public float[] betann { get; set; }

        [LoadColumn(252)]
        public float betaf { get; set; }

        [LoadColumn(253, 265)]
        [VectorType(13)]
        public float[] alphan_betan { get; set; }

        [LoadColumn(266)]
        public string type { get; set; }

        [LoadColumn(267)]
        public float score { get; set; }

        //[LoadColumn(268)]
        //public float discretized_class { get; set; }

        [LoadColumn(269)]
        public int Label { get; set; }
    }
}
