﻿using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace VIC
{
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




    public class ModelInput2
    {
        [ColumnName("fingerprint"), LoadColumn(0)]
        public string Fingerprint { get; set; }


        [ColumnName("minutia"), LoadColumn(1)]
        public string Minutia { get; set; }


        [ColumnName("nn15"), LoadColumn(2)]
        public float Nn15 { get; set; }


        [ColumnName("nn30"), LoadColumn(3)]
        public float Nn30 { get; set; }


        [ColumnName("nn45"), LoadColumn(4)]
        public float Nn45 { get; set; }


        [ColumnName("nn60"), LoadColumn(5)]
        public float Nn60 { get; set; }


        [ColumnName("nn75"), LoadColumn(6)]
        public float Nn75 { get; set; }


        [ColumnName("nn90"), LoadColumn(7)]
        public float Nn90 { get; set; }


        [ColumnName("nn105"), LoadColumn(8)]
        public float Nn105 { get; set; }


        [ColumnName("nn120"), LoadColumn(9)]
        public float Nn120 { get; set; }


        [ColumnName("nn135"), LoadColumn(10)]
        public float Nn135 { get; set; }


        [ColumnName("nn150"), LoadColumn(11)]
        public float Nn150 { get; set; }


        [ColumnName("nn165"), LoadColumn(12)]
        public float Nn165 { get; set; }


        [ColumnName("nn180"), LoadColumn(13)]
        public float Nn180 { get; set; }


        [ColumnName("nn195"), LoadColumn(14)]
        public float Nn195 { get; set; }


        [ColumnName("nn210"), LoadColumn(15)]
        public float Nn210 { get; set; }


        [ColumnName("nn225"), LoadColumn(16)]
        public float Nn225 { get; set; }


        [ColumnName("nn240"), LoadColumn(17)]
        public float Nn240 { get; set; }


        [ColumnName("nn255"), LoadColumn(18)]
        public float Nn255 { get; set; }


        [ColumnName("nn270"), LoadColumn(19)]
        public float Nn270 { get; set; }


        [ColumnName("nn285"), LoadColumn(20)]
        public float Nn285 { get; set; }


        [ColumnName("nn300"), LoadColumn(21)]
        public float Nn300 { get; set; }


        [ColumnName("nn315"), LoadColumn(22)]
        public float Nn315 { get; set; }


        [ColumnName("nn330"), LoadColumn(23)]
        public float Nn330 { get; set; }


        [ColumnName("nn345"), LoadColumn(24)]
        public float Nn345 { get; set; }


        [ColumnName("nn360"), LoadColumn(25)]
        public float Nn360 { get; set; }


        [ColumnName("nn375"), LoadColumn(26)]
        public float Nn375 { get; set; }


        [ColumnName("nn390"), LoadColumn(27)]
        public float Nn390 { get; set; }


        [ColumnName("nn405"), LoadColumn(28)]
        public float Nn405 { get; set; }


        [ColumnName("nn420"), LoadColumn(29)]
        public float Nn420 { get; set; }


        [ColumnName("nn435"), LoadColumn(30)]
        public float Nn435 { get; set; }


        [ColumnName("nn450"), LoadColumn(31)]
        public float Nn450 { get; set; }


        [ColumnName("nn465"), LoadColumn(32)]
        public float Nn465 { get; set; }


        [ColumnName("nn480"), LoadColumn(33)]
        public float Nn480 { get; set; }


        [ColumnName("nn495"), LoadColumn(34)]
        public float Nn495 { get; set; }


        [ColumnName("nn510"), LoadColumn(35)]
        public float Nn510 { get; set; }


        [ColumnName("nn525"), LoadColumn(36)]
        public float Nn525 { get; set; }


        [ColumnName("nn540"), LoadColumn(37)]
        public float Nn540 { get; set; }


        [ColumnName("nn555"), LoadColumn(38)]
        public float Nn555 { get; set; }


        [ColumnName("nn570"), LoadColumn(39)]
        public float Nn570 { get; set; }


        [ColumnName("nn585"), LoadColumn(40)]
        public float Nn585 { get; set; }


        [ColumnName("nn600"), LoadColumn(41)]
        public float Nn600 { get; set; }


        [ColumnName("nn607"), LoadColumn(42)]
        public float Nn607 { get; set; }


        [ColumnName("nn15r"), LoadColumn(43)]
        public float Nn15r { get; set; }


        [ColumnName("nn30r"), LoadColumn(44)]
        public float Nn30r { get; set; }


        [ColumnName("nn45r"), LoadColumn(45)]
        public float Nn45r { get; set; }


        [ColumnName("nn60r"), LoadColumn(46)]
        public float Nn60r { get; set; }


        [ColumnName("nn75r"), LoadColumn(47)]
        public float Nn75r { get; set; }


        [ColumnName("nn90r"), LoadColumn(48)]
        public float Nn90r { get; set; }


        [ColumnName("nn105r"), LoadColumn(49)]
        public float Nn105r { get; set; }


        [ColumnName("nn120r"), LoadColumn(50)]
        public float Nn120r { get; set; }


        [ColumnName("nn135r"), LoadColumn(51)]
        public float Nn135r { get; set; }


        [ColumnName("nn150r"), LoadColumn(52)]
        public float Nn150r { get; set; }


        [ColumnName("nn165r"), LoadColumn(53)]
        public float Nn165r { get; set; }


        [ColumnName("nn180r"), LoadColumn(54)]
        public float Nn180r { get; set; }


        [ColumnName("nn195r"), LoadColumn(55)]
        public float Nn195r { get; set; }


        [ColumnName("nn210r"), LoadColumn(56)]
        public float Nn210r { get; set; }


        [ColumnName("nn225r"), LoadColumn(57)]
        public float Nn225r { get; set; }


        [ColumnName("nn240r"), LoadColumn(58)]
        public float Nn240r { get; set; }


        [ColumnName("nn255r"), LoadColumn(59)]
        public float Nn255r { get; set; }


        [ColumnName("nn270r"), LoadColumn(60)]
        public float Nn270r { get; set; }


        [ColumnName("nn285r"), LoadColumn(61)]
        public float Nn285r { get; set; }


        [ColumnName("nn300r"), LoadColumn(62)]
        public float Nn300r { get; set; }


        [ColumnName("nn315r"), LoadColumn(63)]
        public float Nn315r { get; set; }


        [ColumnName("nn330r"), LoadColumn(64)]
        public float Nn330r { get; set; }


        [ColumnName("nn345r"), LoadColumn(65)]
        public float Nn345r { get; set; }


        [ColumnName("nn360r"), LoadColumn(66)]
        public float Nn360r { get; set; }


        [ColumnName("nn375r"), LoadColumn(67)]
        public float Nn375r { get; set; }


        [ColumnName("nn390r"), LoadColumn(68)]
        public float Nn390r { get; set; }


        [ColumnName("nn405r"), LoadColumn(69)]
        public float Nn405r { get; set; }


        [ColumnName("nn420r"), LoadColumn(70)]
        public float Nn420r { get; set; }


        [ColumnName("nn435r"), LoadColumn(71)]
        public float Nn435r { get; set; }


        [ColumnName("nn450r"), LoadColumn(72)]
        public float Nn450r { get; set; }


        [ColumnName("nn465r"), LoadColumn(73)]
        public float Nn465r { get; set; }


        [ColumnName("nn480r"), LoadColumn(74)]
        public float Nn480r { get; set; }


        [ColumnName("nn495r"), LoadColumn(75)]
        public float Nn495r { get; set; }


        [ColumnName("nn510r"), LoadColumn(76)]
        public float Nn510r { get; set; }


        [ColumnName("nn525r"), LoadColumn(77)]
        public float Nn525r { get; set; }


        [ColumnName("nn540r"), LoadColumn(78)]
        public float Nn540r { get; set; }


        [ColumnName("nn555r"), LoadColumn(79)]
        public float Nn555r { get; set; }


        [ColumnName("nn570r"), LoadColumn(80)]
        public float Nn570r { get; set; }


        [ColumnName("nn585r"), LoadColumn(81)]
        public float Nn585r { get; set; }


        [ColumnName("nn600r"), LoadColumn(82)]
        public float Nn600r { get; set; }


        [ColumnName("nn607r"), LoadColumn(83)]
        public float Nn607r { get; set; }


        [ColumnName("nn30-nn15"), LoadColumn(84)]
        public float Nn30_nn15 { get; set; }


        [ColumnName("nn45-nn30"), LoadColumn(85)]
        public float Nn45_nn30 { get; set; }


        [ColumnName("nn60-nn45"), LoadColumn(86)]
        public float Nn60_nn45 { get; set; }


        [ColumnName("nn75-nn60"), LoadColumn(87)]
        public float Nn75_nn60 { get; set; }


        [ColumnName("nn90-nn75"), LoadColumn(88)]
        public float Nn90_nn75 { get; set; }


        [ColumnName("nn105-nn90"), LoadColumn(89)]
        public float Nn105_nn90 { get; set; }


        [ColumnName("nn120-nn105"), LoadColumn(90)]
        public float Nn120_nn105 { get; set; }


        [ColumnName("nn135-nn120"), LoadColumn(91)]
        public float Nn135_nn120 { get; set; }


        [ColumnName("nn150-nn135"), LoadColumn(92)]
        public float Nn150_nn135 { get; set; }


        [ColumnName("nn165-nn150"), LoadColumn(93)]
        public float Nn165_nn150 { get; set; }


        [ColumnName("nn180-nn165"), LoadColumn(94)]
        public float Nn180_nn165 { get; set; }


        [ColumnName("nn195-nn180"), LoadColumn(95)]
        public float Nn195_nn180 { get; set; }


        [ColumnName("nn210-nn195"), LoadColumn(96)]
        public float Nn210_nn195 { get; set; }


        [ColumnName("nn225-nn210"), LoadColumn(97)]
        public float Nn225_nn210 { get; set; }


        [ColumnName("nn240-nn225"), LoadColumn(98)]
        public float Nn240_nn225 { get; set; }


        [ColumnName("nn255-nn240"), LoadColumn(99)]
        public float Nn255_nn240 { get; set; }


        [ColumnName("nn270-nn255"), LoadColumn(100)]
        public float Nn270_nn255 { get; set; }


        [ColumnName("nn285-nn270"), LoadColumn(101)]
        public float Nn285_nn270 { get; set; }


        [ColumnName("nn300-nn285"), LoadColumn(102)]
        public float Nn300_nn285 { get; set; }


        [ColumnName("nn315-nn300"), LoadColumn(103)]
        public float Nn315_nn300 { get; set; }


        [ColumnName("nn330-nn315"), LoadColumn(104)]
        public float Nn330_nn315 { get; set; }


        [ColumnName("nn345-nn330"), LoadColumn(105)]
        public float Nn345_nn330 { get; set; }


        [ColumnName("nn360-nn345"), LoadColumn(106)]
        public float Nn360_nn345 { get; set; }


        [ColumnName("nn375-nn360"), LoadColumn(107)]
        public float Nn375_nn360 { get; set; }


        [ColumnName("nn390-nn375"), LoadColumn(108)]
        public float Nn390_nn375 { get; set; }


        [ColumnName("nn405-nn390"), LoadColumn(109)]
        public float Nn405_nn390 { get; set; }


        [ColumnName("nn420-nn405"), LoadColumn(110)]
        public float Nn420_nn405 { get; set; }


        [ColumnName("nn435-nn420"), LoadColumn(111)]
        public float Nn435_nn420 { get; set; }


        [ColumnName("nn450-nn435"), LoadColumn(112)]
        public float Nn450_nn435 { get; set; }


        [ColumnName("nn465-nn450"), LoadColumn(113)]
        public float Nn465_nn450 { get; set; }


        [ColumnName("nn480-nn465"), LoadColumn(114)]
        public float Nn480_nn465 { get; set; }


        [ColumnName("nn495-nn480"), LoadColumn(115)]
        public float Nn495_nn480 { get; set; }


        [ColumnName("nn510-nn495"), LoadColumn(116)]
        public float Nn510_nn495 { get; set; }


        [ColumnName("nn525-nn510"), LoadColumn(117)]
        public float Nn525_nn510 { get; set; }


        [ColumnName("nn540-nn525"), LoadColumn(118)]
        public float Nn540_nn525 { get; set; }


        [ColumnName("nn555-nn540"), LoadColumn(119)]
        public float Nn555_nn540 { get; set; }


        [ColumnName("nn570-nn555"), LoadColumn(120)]
        public float Nn570_nn555 { get; set; }


        [ColumnName("nn585-nn570"), LoadColumn(121)]
        public float Nn585_nn570 { get; set; }


        [ColumnName("nn600-nn585"), LoadColumn(122)]
        public float Nn600_nn585 { get; set; }


        [ColumnName("nn607-nn600"), LoadColumn(123)]
        public float Nn607_nn600 { get; set; }


        [ColumnName("nn30r-nn15r"), LoadColumn(124)]
        public float Nn30r_nn15r { get; set; }


        [ColumnName("nn45r-nn30r"), LoadColumn(125)]
        public float Nn45r_nn30r { get; set; }


        [ColumnName("nn60r-nn45r"), LoadColumn(126)]
        public float Nn60r_nn45r { get; set; }


        [ColumnName("nn75r-nn60r"), LoadColumn(127)]
        public float Nn75r_nn60r { get; set; }


        [ColumnName("nn90r-nn75r"), LoadColumn(128)]
        public float Nn90r_nn75r { get; set; }


        [ColumnName("nn105r-nn90r"), LoadColumn(129)]
        public float Nn105r_nn90r { get; set; }


        [ColumnName("nn120r-nn105r"), LoadColumn(130)]
        public float Nn120r_nn105r { get; set; }


        [ColumnName("nn135r-nn120r"), LoadColumn(131)]
        public float Nn135r_nn120r { get; set; }


        [ColumnName("nn150r-nn135r"), LoadColumn(132)]
        public float Nn150r_nn135r { get; set; }


        [ColumnName("nn165r-nn150r"), LoadColumn(133)]
        public float Nn165r_nn150r { get; set; }


        [ColumnName("nn180r-nn165r"), LoadColumn(134)]
        public float Nn180r_nn165r { get; set; }


        [ColumnName("nn195r-nn180r"), LoadColumn(135)]
        public float Nn195r_nn180r { get; set; }


        [ColumnName("nn210r-nn195r"), LoadColumn(136)]
        public float Nn210r_nn195r { get; set; }


        [ColumnName("nn225r-nn210r"), LoadColumn(137)]
        public float Nn225r_nn210r { get; set; }


        [ColumnName("nn240r-nn225r"), LoadColumn(138)]
        public float Nn240r_nn225r { get; set; }


        [ColumnName("nn255r-nn240r"), LoadColumn(139)]
        public float Nn255r_nn240r { get; set; }


        [ColumnName("nn270r-nn255r"), LoadColumn(140)]
        public float Nn270r_nn255r { get; set; }


        [ColumnName("nn285r-nn270r"), LoadColumn(141)]
        public float Nn285r_nn270r { get; set; }


        [ColumnName("nn300r-nn285r"), LoadColumn(142)]
        public float Nn300r_nn285r { get; set; }


        [ColumnName("nn315r-nn300r"), LoadColumn(143)]
        public float Nn315r_nn300r { get; set; }


        [ColumnName("nn330r-nn315r"), LoadColumn(144)]
        public float Nn330r_nn315r { get; set; }


        [ColumnName("nn345r-nn330r"), LoadColumn(145)]
        public float Nn345r_nn330r { get; set; }


        [ColumnName("nn360r-nn345r"), LoadColumn(146)]
        public float Nn360r_nn345r { get; set; }


        [ColumnName("nn375r-nn360r"), LoadColumn(147)]
        public float Nn375r_nn360r { get; set; }


        [ColumnName("nn390r-nn375r"), LoadColumn(148)]
        public float Nn390r_nn375r { get; set; }


        [ColumnName("nn405r-nn390r"), LoadColumn(149)]
        public float Nn405r_nn390r { get; set; }


        [ColumnName("nn420r-nn405r"), LoadColumn(150)]
        public float Nn420r_nn405r { get; set; }


        [ColumnName("nn435r-nn420r"), LoadColumn(151)]
        public float Nn435r_nn420r { get; set; }


        [ColumnName("nn450r-nn435r"), LoadColumn(152)]
        public float Nn450r_nn435r { get; set; }


        [ColumnName("nn465r-nn450r"), LoadColumn(153)]
        public float Nn465r_nn450r { get; set; }


        [ColumnName("nn480r-nn465r"), LoadColumn(154)]
        public float Nn480r_nn465r { get; set; }


        [ColumnName("nn495r-nn480r"), LoadColumn(155)]
        public float Nn495r_nn480r { get; set; }


        [ColumnName("nn510r-nn495r"), LoadColumn(156)]
        public float Nn510r_nn495r { get; set; }


        [ColumnName("nn525r-nn510r"), LoadColumn(157)]
        public float Nn525r_nn510r { get; set; }


        [ColumnName("nn540r-nn525r"), LoadColumn(158)]
        public float Nn540r_nn525r { get; set; }


        [ColumnName("nn555r-nn540r"), LoadColumn(159)]
        public float Nn555r_nn540r { get; set; }


        [ColumnName("nn570r-nn555r"), LoadColumn(160)]
        public float Nn570r_nn555r { get; set; }


        [ColumnName("nn585r-nn570r"), LoadColumn(161)]
        public float Nn585r_nn570r { get; set; }


        [ColumnName("nn600r-nn585r"), LoadColumn(162)]
        public float Nn600r_nn585r { get; set; }


        [ColumnName("nn607r-nn600r"), LoadColumn(163)]
        public float Nn607r_nn600r { get; set; }


        [ColumnName("d1"), LoadColumn(164)]
        public float D1 { get; set; }


        [ColumnName("d2"), LoadColumn(165)]
        public float D2 { get; set; }


        [ColumnName("d3"), LoadColumn(166)]
        public float D3 { get; set; }


        [ColumnName("d4"), LoadColumn(167)]
        public float D4 { get; set; }


        [ColumnName("d5"), LoadColumn(168)]
        public float D5 { get; set; }


        [ColumnName("d6"), LoadColumn(169)]
        public float D6 { get; set; }


        [ColumnName("d7"), LoadColumn(170)]
        public float D7 { get; set; }


        [ColumnName("d8"), LoadColumn(171)]
        public float D8 { get; set; }


        [ColumnName("d9"), LoadColumn(172)]
        public float D9 { get; set; }


        [ColumnName("d10"), LoadColumn(173)]
        public float D10 { get; set; }


        [ColumnName("d11"), LoadColumn(174)]
        public float D11 { get; set; }


        [ColumnName("d12"), LoadColumn(175)]
        public float D12 { get; set; }


        [ColumnName("df"), LoadColumn(176)]
        public float Df { get; set; }


        [ColumnName("d1r"), LoadColumn(177)]
        public float D1r { get; set; }


        [ColumnName("d2r"), LoadColumn(178)]
        public float D2r { get; set; }


        [ColumnName("d3r"), LoadColumn(179)]
        public float D3r { get; set; }


        [ColumnName("d4r"), LoadColumn(180)]
        public float D4r { get; set; }


        [ColumnName("d5r"), LoadColumn(181)]
        public float D5r { get; set; }


        [ColumnName("d6r"), LoadColumn(182)]
        public float D6r { get; set; }


        [ColumnName("d7r"), LoadColumn(183)]
        public float D7r { get; set; }


        [ColumnName("d8r"), LoadColumn(184)]
        public float D8r { get; set; }


        [ColumnName("d9r"), LoadColumn(185)]
        public float D9r { get; set; }


        [ColumnName("d10r"), LoadColumn(186)]
        public float D10r { get; set; }


        [ColumnName("d11r"), LoadColumn(187)]
        public float D11r { get; set; }


        [ColumnName("d12r"), LoadColumn(188)]
        public float D12r { get; set; }


        [ColumnName("dfr"), LoadColumn(189)]
        public float Dfr { get; set; }


        [ColumnName("d2-d1"), LoadColumn(190)]
        public float D2_d1 { get; set; }


        [ColumnName("d3-d2"), LoadColumn(191)]
        public float D3_d2 { get; set; }


        [ColumnName("d4-d3"), LoadColumn(192)]
        public float D4_d3 { get; set; }


        [ColumnName("d5-d4"), LoadColumn(193)]
        public float D5_d4 { get; set; }


        [ColumnName("d6-d5"), LoadColumn(194)]
        public float D6_d5 { get; set; }


        [ColumnName("d7-d6"), LoadColumn(195)]
        public float D7_d6 { get; set; }


        [ColumnName("d8-d7"), LoadColumn(196)]
        public float D8_d7 { get; set; }


        [ColumnName("d9-d8"), LoadColumn(197)]
        public float D9_d8 { get; set; }


        [ColumnName("d10-d9"), LoadColumn(198)]
        public float D10_d9 { get; set; }


        [ColumnName("d11-d10"), LoadColumn(199)]
        public float D11_d10 { get; set; }


        [ColumnName("d12-d11"), LoadColumn(200)]
        public float D12_d11 { get; set; }


        [ColumnName("df-d12"), LoadColumn(201)]
        public float Df_d12 { get; set; }


        [ColumnName("d2r-d1r"), LoadColumn(202)]
        public float D2r_d1r { get; set; }


        [ColumnName("d3r-d2r"), LoadColumn(203)]
        public float D3r_d2r { get; set; }


        [ColumnName("d4r-d3r"), LoadColumn(204)]
        public float D4r_d3r { get; set; }


        [ColumnName("d5r-d4r"), LoadColumn(205)]
        public float D5r_d4r { get; set; }


        [ColumnName("d6r-d5r"), LoadColumn(206)]
        public float D6r_d5r { get; set; }


        [ColumnName("d7r-d6r"), LoadColumn(207)]
        public float D7r_d6r { get; set; }


        [ColumnName("d8r-d7r"), LoadColumn(208)]
        public float D8r_d7r { get; set; }


        [ColumnName("d9r-d8r"), LoadColumn(209)]
        public float D9r_d8r { get; set; }


        [ColumnName("d10r-d9r"), LoadColumn(210)]
        public float D10r_d9r { get; set; }


        [ColumnName("d11r-d10r"), LoadColumn(211)]
        public float D11r_d10r { get; set; }


        [ColumnName("d12r-d11r"), LoadColumn(212)]
        public float D12r_d11r { get; set; }


        [ColumnName("dfr-d12r"), LoadColumn(213)]
        public float Dfr_d12r { get; set; }


        [ColumnName("alpha1"), LoadColumn(214)]
        public float Alpha1 { get; set; }


        [ColumnName("alpha2"), LoadColumn(215)]
        public float Alpha2 { get; set; }


        [ColumnName("alpha3"), LoadColumn(216)]
        public float Alpha3 { get; set; }


        [ColumnName("alpha4"), LoadColumn(217)]
        public float Alpha4 { get; set; }


        [ColumnName("alpha5"), LoadColumn(218)]
        public float Alpha5 { get; set; }


        [ColumnName("alpha6"), LoadColumn(219)]
        public float Alpha6 { get; set; }


        [ColumnName("alpha7"), LoadColumn(220)]
        public float Alpha7 { get; set; }


        [ColumnName("alpha8"), LoadColumn(221)]
        public float Alpha8 { get; set; }


        [ColumnName("alpha9"), LoadColumn(222)]
        public float Alpha9 { get; set; }


        [ColumnName("alpha10"), LoadColumn(223)]
        public float Alpha10 { get; set; }


        [ColumnName("alpha11"), LoadColumn(224)]
        public float Alpha11 { get; set; }


        [ColumnName("alpha12"), LoadColumn(225)]
        public float Alpha12 { get; set; }


        [ColumnName("alphaf"), LoadColumn(226)]
        public float Alphaf { get; set; }


        [ColumnName("alphan1"), LoadColumn(227)]
        public float Alphan1 { get; set; }


        [ColumnName("alphan2"), LoadColumn(228)]
        public float Alphan2 { get; set; }


        [ColumnName("alphan3"), LoadColumn(229)]
        public float Alphan3 { get; set; }


        [ColumnName("alphan4"), LoadColumn(230)]
        public float Alphan4 { get; set; }


        [ColumnName("alphan5"), LoadColumn(231)]
        public float Alphan5 { get; set; }


        [ColumnName("alphan6"), LoadColumn(232)]
        public float Alphan6 { get; set; }


        [ColumnName("alphan7"), LoadColumn(233)]
        public float Alphan7 { get; set; }


        [ColumnName("alphan8"), LoadColumn(234)]
        public float Alphan8 { get; set; }


        [ColumnName("alphan9"), LoadColumn(235)]
        public float Alphan9 { get; set; }


        [ColumnName("alphan10"), LoadColumn(236)]
        public float Alphan10 { get; set; }


        [ColumnName("alphan11"), LoadColumn(237)]
        public float Alphan11 { get; set; }


        [ColumnName("alphan12"), LoadColumn(238)]
        public float Alphan12 { get; set; }


        [ColumnName("alphanf"), LoadColumn(239)]
        public float Alphanf { get; set; }


        [ColumnName("beta1"), LoadColumn(240)]
        public float Beta1 { get; set; }


        [ColumnName("beta2"), LoadColumn(241)]
        public float Beta2 { get; set; }


        [ColumnName("beta3"), LoadColumn(242)]
        public float Beta3 { get; set; }


        [ColumnName("beta4"), LoadColumn(243)]
        public float Beta4 { get; set; }


        [ColumnName("beta5"), LoadColumn(244)]
        public float Beta5 { get; set; }


        [ColumnName("beta6"), LoadColumn(245)]
        public float Beta6 { get; set; }


        [ColumnName("beta7"), LoadColumn(246)]
        public float Beta7 { get; set; }


        [ColumnName("beta8"), LoadColumn(247)]
        public float Beta8 { get; set; }


        [ColumnName("beta9"), LoadColumn(248)]
        public float Beta9 { get; set; }


        [ColumnName("beta10"), LoadColumn(249)]
        public float Beta10 { get; set; }


        [ColumnName("beta11"), LoadColumn(250)]
        public float Beta11 { get; set; }


        [ColumnName("beta12"), LoadColumn(251)]
        public float Beta12 { get; set; }


        [ColumnName("betaf"), LoadColumn(252)]
        public float Betaf { get; set; }


        [ColumnName("alpha1-beta1"), LoadColumn(253)]
        public float Alpha1_beta1 { get; set; }


        [ColumnName("alpha2-beta2"), LoadColumn(254)]
        public float Alpha2_beta2 { get; set; }


        [ColumnName("alpha3-beta3"), LoadColumn(255)]
        public float Alpha3_beta3 { get; set; }


        [ColumnName("alpha4-beta4"), LoadColumn(256)]
        public float Alpha4_beta4 { get; set; }


        [ColumnName("alpha5-beta5"), LoadColumn(257)]
        public float Alpha5_beta5 { get; set; }


        [ColumnName("alpha6-beta6"), LoadColumn(258)]
        public float Alpha6_beta6 { get; set; }


        [ColumnName("alpha7-beta7"), LoadColumn(259)]
        public float Alpha7_beta7 { get; set; }


        [ColumnName("alpha8-beta8"), LoadColumn(260)]
        public float Alpha8_beta8 { get; set; }


        [ColumnName("alpha9-beta9"), LoadColumn(261)]
        public float Alpha9_beta9 { get; set; }


        [ColumnName("alpha10-beta10"), LoadColumn(262)]
        public float Alpha10_beta10 { get; set; }


        [ColumnName("alpha11-beta11"), LoadColumn(263)]
        public float Alpha11_beta11 { get; set; }


        [ColumnName("alpha12-beta12"), LoadColumn(264)]
        public float Alpha12_beta12 { get; set; }


        [ColumnName("alphaf-betaf"), LoadColumn(265)]
        public float Alphaf_betaf { get; set; }


        [ColumnName("type"), LoadColumn(266)]
        public string Type { get; set; }


        [ColumnName("score_val"), LoadColumn(267)]
        public float Score_val { get; set; }


        [ColumnName("Label"), LoadColumn(268)]
        public int Label { get; set; }




    }

    public class ModelInput
    {
        [ColumnName("fingerprint"), LoadColumn(0)]
        public string Fingerprint { get; set; }


        [ColumnName("minutia"), LoadColumn(1)]
        public string Minutia { get; set; }


        [ColumnName("nn15"), LoadColumn(2)]
        public float Nn15 { get; set; }


        [ColumnName("nn30"), LoadColumn(3)]
        public float Nn30 { get; set; }


        [ColumnName("nn45"), LoadColumn(4)]
        public float Nn45 { get; set; }


        [ColumnName("nn60"), LoadColumn(5)]
        public float Nn60 { get; set; }


        [ColumnName("nn75"), LoadColumn(6)]
        public float Nn75 { get; set; }


        [ColumnName("nn90"), LoadColumn(7)]
        public float Nn90 { get; set; }


        [ColumnName("nn105"), LoadColumn(8)]
        public float Nn105 { get; set; }


        [ColumnName("nn120"), LoadColumn(9)]
        public float Nn120 { get; set; }


        [ColumnName("nn135"), LoadColumn(10)]
        public float Nn135 { get; set; }


        [ColumnName("nn150"), LoadColumn(11)]
        public float Nn150 { get; set; }


        [ColumnName("nn165"), LoadColumn(12)]
        public float Nn165 { get; set; }


        [ColumnName("nn180"), LoadColumn(13)]
        public float Nn180 { get; set; }


        [ColumnName("nn195"), LoadColumn(14)]
        public float Nn195 { get; set; }


        [ColumnName("nn210"), LoadColumn(15)]
        public float Nn210 { get; set; }


        [ColumnName("nn225"), LoadColumn(16)]
        public float Nn225 { get; set; }


        [ColumnName("nn240"), LoadColumn(17)]
        public float Nn240 { get; set; }


        [ColumnName("nn255"), LoadColumn(18)]
        public float Nn255 { get; set; }


        [ColumnName("nn270"), LoadColumn(19)]
        public float Nn270 { get; set; }


        [ColumnName("nn285"), LoadColumn(20)]
        public float Nn285 { get; set; }


        [ColumnName("nn300"), LoadColumn(21)]
        public float Nn300 { get; set; }


        [ColumnName("nn315"), LoadColumn(22)]
        public float Nn315 { get; set; }


        [ColumnName("nn330"), LoadColumn(23)]
        public float Nn330 { get; set; }


        [ColumnName("nn345"), LoadColumn(24)]
        public float Nn345 { get; set; }


        [ColumnName("nn360"), LoadColumn(25)]
        public float Nn360 { get; set; }


        [ColumnName("nn375"), LoadColumn(26)]
        public float Nn375 { get; set; }


        [ColumnName("nn390"), LoadColumn(27)]
        public float Nn390 { get; set; }


        [ColumnName("nn405"), LoadColumn(28)]
        public float Nn405 { get; set; }


        [ColumnName("nn420"), LoadColumn(29)]
        public float Nn420 { get; set; }


        [ColumnName("nn435"), LoadColumn(30)]
        public float Nn435 { get; set; }


        [ColumnName("nn450"), LoadColumn(31)]
        public float Nn450 { get; set; }


        [ColumnName("nn465"), LoadColumn(32)]
        public float Nn465 { get; set; }


        [ColumnName("nn480"), LoadColumn(33)]
        public float Nn480 { get; set; }


        [ColumnName("nn495"), LoadColumn(34)]
        public float Nn495 { get; set; }


        [ColumnName("nn510"), LoadColumn(35)]
        public float Nn510 { get; set; }


        [ColumnName("nn525"), LoadColumn(36)]
        public float Nn525 { get; set; }


        [ColumnName("nn540"), LoadColumn(37)]
        public float Nn540 { get; set; }


        [ColumnName("nn555"), LoadColumn(38)]
        public float Nn555 { get; set; }


        [ColumnName("nn570"), LoadColumn(39)]
        public float Nn570 { get; set; }


        [ColumnName("nn585"), LoadColumn(40)]
        public float Nn585 { get; set; }


        [ColumnName("nn600"), LoadColumn(41)]
        public float Nn600 { get; set; }


        [ColumnName("nn607"), LoadColumn(42)]
        public float Nn607 { get; set; }


        [ColumnName("nn15r"), LoadColumn(43)]
        public float Nn15r { get; set; }


        [ColumnName("nn30r"), LoadColumn(44)]
        public float Nn30r { get; set; }


        [ColumnName("nn45r"), LoadColumn(45)]
        public float Nn45r { get; set; }


        [ColumnName("nn60r"), LoadColumn(46)]
        public float Nn60r { get; set; }


        [ColumnName("nn75r"), LoadColumn(47)]
        public float Nn75r { get; set; }


        [ColumnName("nn90r"), LoadColumn(48)]
        public float Nn90r { get; set; }


        [ColumnName("nn105r"), LoadColumn(49)]
        public float Nn105r { get; set; }


        [ColumnName("nn120r"), LoadColumn(50)]
        public float Nn120r { get; set; }


        [ColumnName("nn135r"), LoadColumn(51)]
        public float Nn135r { get; set; }


        [ColumnName("nn150r"), LoadColumn(52)]
        public float Nn150r { get; set; }


        [ColumnName("nn165r"), LoadColumn(53)]
        public float Nn165r { get; set; }


        [ColumnName("nn180r"), LoadColumn(54)]
        public float Nn180r { get; set; }


        [ColumnName("nn195r"), LoadColumn(55)]
        public float Nn195r { get; set; }


        [ColumnName("nn210r"), LoadColumn(56)]
        public float Nn210r { get; set; }


        [ColumnName("nn225r"), LoadColumn(57)]
        public float Nn225r { get; set; }


        [ColumnName("nn240r"), LoadColumn(58)]
        public float Nn240r { get; set; }


        [ColumnName("nn255r"), LoadColumn(59)]
        public float Nn255r { get; set; }


        [ColumnName("nn270r"), LoadColumn(60)]
        public float Nn270r { get; set; }


        [ColumnName("nn285r"), LoadColumn(61)]
        public float Nn285r { get; set; }


        [ColumnName("nn300r"), LoadColumn(62)]
        public float Nn300r { get; set; }


        [ColumnName("nn315r"), LoadColumn(63)]
        public float Nn315r { get; set; }


        [ColumnName("nn330r"), LoadColumn(64)]
        public float Nn330r { get; set; }


        [ColumnName("nn345r"), LoadColumn(65)]
        public float Nn345r { get; set; }


        [ColumnName("nn360r"), LoadColumn(66)]
        public float Nn360r { get; set; }


        [ColumnName("nn375r"), LoadColumn(67)]
        public float Nn375r { get; set; }


        [ColumnName("nn390r"), LoadColumn(68)]
        public float Nn390r { get; set; }


        [ColumnName("nn405r"), LoadColumn(69)]
        public float Nn405r { get; set; }


        [ColumnName("nn420r"), LoadColumn(70)]
        public float Nn420r { get; set; }


        [ColumnName("nn435r"), LoadColumn(71)]
        public float Nn435r { get; set; }


        [ColumnName("nn450r"), LoadColumn(72)]
        public float Nn450r { get; set; }


        [ColumnName("nn465r"), LoadColumn(73)]
        public float Nn465r { get; set; }


        [ColumnName("nn480r"), LoadColumn(74)]
        public float Nn480r { get; set; }


        [ColumnName("nn495r"), LoadColumn(75)]
        public float Nn495r { get; set; }


        [ColumnName("nn510r"), LoadColumn(76)]
        public float Nn510r { get; set; }


        [ColumnName("nn525r"), LoadColumn(77)]
        public float Nn525r { get; set; }


        [ColumnName("nn540r"), LoadColumn(78)]
        public float Nn540r { get; set; }


        [ColumnName("nn555r"), LoadColumn(79)]
        public float Nn555r { get; set; }


        [ColumnName("nn570r"), LoadColumn(80)]
        public float Nn570r { get; set; }


        [ColumnName("nn585r"), LoadColumn(81)]
        public float Nn585r { get; set; }


        [ColumnName("nn600r"), LoadColumn(82)]
        public float Nn600r { get; set; }


        [ColumnName("nn607r"), LoadColumn(83)]
        public float Nn607r { get; set; }


        [ColumnName("nn30-nn15"), LoadColumn(84)]
        public float Nn30_nn15 { get; set; }


        [ColumnName("nn45-nn30"), LoadColumn(85)]
        public float Nn45_nn30 { get; set; }


        [ColumnName("nn60-nn45"), LoadColumn(86)]
        public float Nn60_nn45 { get; set; }


        [ColumnName("nn75-nn60"), LoadColumn(87)]
        public float Nn75_nn60 { get; set; }


        [ColumnName("nn90-nn75"), LoadColumn(88)]
        public float Nn90_nn75 { get; set; }


        [ColumnName("nn105-nn90"), LoadColumn(89)]
        public float Nn105_nn90 { get; set; }


        [ColumnName("nn120-nn105"), LoadColumn(90)]
        public float Nn120_nn105 { get; set; }


        [ColumnName("nn135-nn120"), LoadColumn(91)]
        public float Nn135_nn120 { get; set; }


        [ColumnName("nn150-nn135"), LoadColumn(92)]
        public float Nn150_nn135 { get; set; }


        [ColumnName("nn165-nn150"), LoadColumn(93)]
        public float Nn165_nn150 { get; set; }


        [ColumnName("nn180-nn165"), LoadColumn(94)]
        public float Nn180_nn165 { get; set; }


        [ColumnName("nn195-nn180"), LoadColumn(95)]
        public float Nn195_nn180 { get; set; }


        [ColumnName("nn210-nn195"), LoadColumn(96)]
        public float Nn210_nn195 { get; set; }


        [ColumnName("nn225-nn210"), LoadColumn(97)]
        public float Nn225_nn210 { get; set; }


        [ColumnName("nn240-nn225"), LoadColumn(98)]
        public float Nn240_nn225 { get; set; }


        [ColumnName("nn255-nn240"), LoadColumn(99)]
        public float Nn255_nn240 { get; set; }


        [ColumnName("nn270-nn255"), LoadColumn(100)]
        public float Nn270_nn255 { get; set; }


        [ColumnName("nn285-nn270"), LoadColumn(101)]
        public float Nn285_nn270 { get; set; }


        [ColumnName("nn300-nn285"), LoadColumn(102)]
        public float Nn300_nn285 { get; set; }


        [ColumnName("nn315-nn300"), LoadColumn(103)]
        public float Nn315_nn300 { get; set; }


        [ColumnName("nn330-nn315"), LoadColumn(104)]
        public float Nn330_nn315 { get; set; }


        [ColumnName("nn345-nn330"), LoadColumn(105)]
        public float Nn345_nn330 { get; set; }


        [ColumnName("nn360-nn345"), LoadColumn(106)]
        public float Nn360_nn345 { get; set; }


        [ColumnName("nn375-nn360"), LoadColumn(107)]
        public float Nn375_nn360 { get; set; }


        [ColumnName("nn390-nn375"), LoadColumn(108)]
        public float Nn390_nn375 { get; set; }


        [ColumnName("nn405-nn390"), LoadColumn(109)]
        public float Nn405_nn390 { get; set; }


        [ColumnName("nn420-nn405"), LoadColumn(110)]
        public float Nn420_nn405 { get; set; }


        [ColumnName("nn435-nn420"), LoadColumn(111)]
        public float Nn435_nn420 { get; set; }


        [ColumnName("nn450-nn435"), LoadColumn(112)]
        public float Nn450_nn435 { get; set; }


        [ColumnName("nn465-nn450"), LoadColumn(113)]
        public float Nn465_nn450 { get; set; }


        [ColumnName("nn480-nn465"), LoadColumn(114)]
        public float Nn480_nn465 { get; set; }


        [ColumnName("nn495-nn480"), LoadColumn(115)]
        public float Nn495_nn480 { get; set; }


        [ColumnName("nn510-nn495"), LoadColumn(116)]
        public float Nn510_nn495 { get; set; }


        [ColumnName("nn525-nn510"), LoadColumn(117)]
        public float Nn525_nn510 { get; set; }


        [ColumnName("nn540-nn525"), LoadColumn(118)]
        public float Nn540_nn525 { get; set; }


        [ColumnName("nn555-nn540"), LoadColumn(119)]
        public float Nn555_nn540 { get; set; }


        [ColumnName("nn570-nn555"), LoadColumn(120)]
        public float Nn570_nn555 { get; set; }


        [ColumnName("nn585-nn570"), LoadColumn(121)]
        public float Nn585_nn570 { get; set; }


        [ColumnName("nn600-nn585"), LoadColumn(122)]
        public float Nn600_nn585 { get; set; }


        [ColumnName("nn607-nn600"), LoadColumn(123)]
        public float Nn607_nn600 { get; set; }


        [ColumnName("nn30r-nn15r"), LoadColumn(124)]
        public float Nn30r_nn15r { get; set; }


        [ColumnName("nn45r-nn30r"), LoadColumn(125)]
        public float Nn45r_nn30r { get; set; }


        [ColumnName("nn60r-nn45r"), LoadColumn(126)]
        public float Nn60r_nn45r { get; set; }


        [ColumnName("nn75r-nn60r"), LoadColumn(127)]
        public float Nn75r_nn60r { get; set; }


        [ColumnName("nn90r-nn75r"), LoadColumn(128)]
        public float Nn90r_nn75r { get; set; }


        [ColumnName("nn105r-nn90r"), LoadColumn(129)]
        public float Nn105r_nn90r { get; set; }


        [ColumnName("nn120r-nn105r"), LoadColumn(130)]
        public float Nn120r_nn105r { get; set; }


        [ColumnName("nn135r-nn120r"), LoadColumn(131)]
        public float Nn135r_nn120r { get; set; }


        [ColumnName("nn150r-nn135r"), LoadColumn(132)]
        public float Nn150r_nn135r { get; set; }


        [ColumnName("nn165r-nn150r"), LoadColumn(133)]
        public float Nn165r_nn150r { get; set; }


        [ColumnName("nn180r-nn165r"), LoadColumn(134)]
        public float Nn180r_nn165r { get; set; }


        [ColumnName("nn195r-nn180r"), LoadColumn(135)]
        public float Nn195r_nn180r { get; set; }


        [ColumnName("nn210r-nn195r"), LoadColumn(136)]
        public float Nn210r_nn195r { get; set; }


        [ColumnName("nn225r-nn210r"), LoadColumn(137)]
        public float Nn225r_nn210r { get; set; }


        [ColumnName("nn240r-nn225r"), LoadColumn(138)]
        public float Nn240r_nn225r { get; set; }


        [ColumnName("nn255r-nn240r"), LoadColumn(139)]
        public float Nn255r_nn240r { get; set; }


        [ColumnName("nn270r-nn255r"), LoadColumn(140)]
        public float Nn270r_nn255r { get; set; }


        [ColumnName("nn285r-nn270r"), LoadColumn(141)]
        public float Nn285r_nn270r { get; set; }


        [ColumnName("nn300r-nn285r"), LoadColumn(142)]
        public float Nn300r_nn285r { get; set; }


        [ColumnName("nn315r-nn300r"), LoadColumn(143)]
        public float Nn315r_nn300r { get; set; }


        [ColumnName("nn330r-nn315r"), LoadColumn(144)]
        public float Nn330r_nn315r { get; set; }


        [ColumnName("nn345r-nn330r"), LoadColumn(145)]
        public float Nn345r_nn330r { get; set; }


        [ColumnName("nn360r-nn345r"), LoadColumn(146)]
        public float Nn360r_nn345r { get; set; }


        [ColumnName("nn375r-nn360r"), LoadColumn(147)]
        public float Nn375r_nn360r { get; set; }


        [ColumnName("nn390r-nn375r"), LoadColumn(148)]
        public float Nn390r_nn375r { get; set; }


        [ColumnName("nn405r-nn390r"), LoadColumn(149)]
        public float Nn405r_nn390r { get; set; }


        [ColumnName("nn420r-nn405r"), LoadColumn(150)]
        public float Nn420r_nn405r { get; set; }


        [ColumnName("nn435r-nn420r"), LoadColumn(151)]
        public float Nn435r_nn420r { get; set; }


        [ColumnName("nn450r-nn435r"), LoadColumn(152)]
        public float Nn450r_nn435r { get; set; }


        [ColumnName("nn465r-nn450r"), LoadColumn(153)]
        public float Nn465r_nn450r { get; set; }


        [ColumnName("nn480r-nn465r"), LoadColumn(154)]
        public float Nn480r_nn465r { get; set; }


        [ColumnName("nn495r-nn480r"), LoadColumn(155)]
        public float Nn495r_nn480r { get; set; }


        [ColumnName("nn510r-nn495r"), LoadColumn(156)]
        public float Nn510r_nn495r { get; set; }


        [ColumnName("nn525r-nn510r"), LoadColumn(157)]
        public float Nn525r_nn510r { get; set; }


        [ColumnName("nn540r-nn525r"), LoadColumn(158)]
        public float Nn540r_nn525r { get; set; }


        [ColumnName("nn555r-nn540r"), LoadColumn(159)]
        public float Nn555r_nn540r { get; set; }


        [ColumnName("nn570r-nn555r"), LoadColumn(160)]
        public float Nn570r_nn555r { get; set; }


        [ColumnName("nn585r-nn570r"), LoadColumn(161)]
        public float Nn585r_nn570r { get; set; }


        [ColumnName("nn600r-nn585r"), LoadColumn(162)]
        public float Nn600r_nn585r { get; set; }


        [ColumnName("nn607r-nn600r"), LoadColumn(163)]
        public float Nn607r_nn600r { get; set; }


        [ColumnName("d1"), LoadColumn(164)]
        public float D1 { get; set; }


        [ColumnName("d2"), LoadColumn(165)]
        public float D2 { get; set; }


        [ColumnName("d3"), LoadColumn(166)]
        public float D3 { get; set; }


        [ColumnName("d4"), LoadColumn(167)]
        public float D4 { get; set; }


        [ColumnName("d5"), LoadColumn(168)]
        public float D5 { get; set; }


        [ColumnName("d6"), LoadColumn(169)]
        public float D6 { get; set; }


        [ColumnName("d7"), LoadColumn(170)]
        public float D7 { get; set; }


        [ColumnName("d8"), LoadColumn(171)]
        public float D8 { get; set; }


        [ColumnName("d9"), LoadColumn(172)]
        public float D9 { get; set; }


        [ColumnName("d10"), LoadColumn(173)]
        public float D10 { get; set; }


        [ColumnName("d11"), LoadColumn(174)]
        public float D11 { get; set; }


        [ColumnName("d12"), LoadColumn(175)]
        public float D12 { get; set; }


        [ColumnName("df"), LoadColumn(176)]
        public float Df { get; set; }


        [ColumnName("d1r"), LoadColumn(177)]
        public float D1r { get; set; }


        [ColumnName("d2r"), LoadColumn(178)]
        public float D2r { get; set; }


        [ColumnName("d3r"), LoadColumn(179)]
        public float D3r { get; set; }


        [ColumnName("d4r"), LoadColumn(180)]
        public float D4r { get; set; }


        [ColumnName("d5r"), LoadColumn(181)]
        public float D5r { get; set; }


        [ColumnName("d6r"), LoadColumn(182)]
        public float D6r { get; set; }


        [ColumnName("d7r"), LoadColumn(183)]
        public float D7r { get; set; }


        [ColumnName("d8r"), LoadColumn(184)]
        public float D8r { get; set; }


        [ColumnName("d9r"), LoadColumn(185)]
        public float D9r { get; set; }


        [ColumnName("d10r"), LoadColumn(186)]
        public float D10r { get; set; }


        [ColumnName("d11r"), LoadColumn(187)]
        public float D11r { get; set; }


        [ColumnName("d12r"), LoadColumn(188)]
        public float D12r { get; set; }


        [ColumnName("dfr"), LoadColumn(189)]
        public float Dfr { get; set; }


        [ColumnName("d2-d1"), LoadColumn(190)]
        public float D2_d1 { get; set; }


        [ColumnName("d3-d2"), LoadColumn(191)]
        public float D3_d2 { get; set; }


        [ColumnName("d4-d3"), LoadColumn(192)]
        public float D4_d3 { get; set; }


        [ColumnName("d5-d4"), LoadColumn(193)]
        public float D5_d4 { get; set; }


        [ColumnName("d6-d5"), LoadColumn(194)]
        public float D6_d5 { get; set; }


        [ColumnName("d7-d6"), LoadColumn(195)]
        public float D7_d6 { get; set; }


        [ColumnName("d8-d7"), LoadColumn(196)]
        public float D8_d7 { get; set; }


        [ColumnName("d9-d8"), LoadColumn(197)]
        public float D9_d8 { get; set; }


        [ColumnName("d10-d9"), LoadColumn(198)]
        public float D10_d9 { get; set; }


        [ColumnName("d11-d10"), LoadColumn(199)]
        public float D11_d10 { get; set; }


        [ColumnName("d12-d11"), LoadColumn(200)]
        public float D12_d11 { get; set; }


        [ColumnName("df-d12"), LoadColumn(201)]
        public float Df_d12 { get; set; }


        [ColumnName("d2r-d1r"), LoadColumn(202)]
        public float D2r_d1r { get; set; }


        [ColumnName("d3r-d2r"), LoadColumn(203)]
        public float D3r_d2r { get; set; }


        [ColumnName("d4r-d3r"), LoadColumn(204)]
        public float D4r_d3r { get; set; }


        [ColumnName("d5r-d4r"), LoadColumn(205)]
        public float D5r_d4r { get; set; }


        [ColumnName("d6r-d5r"), LoadColumn(206)]
        public float D6r_d5r { get; set; }


        [ColumnName("d7r-d6r"), LoadColumn(207)]
        public float D7r_d6r { get; set; }


        [ColumnName("d8r-d7r"), LoadColumn(208)]
        public float D8r_d7r { get; set; }


        [ColumnName("d9r-d8r"), LoadColumn(209)]
        public float D9r_d8r { get; set; }


        [ColumnName("d10r-d9r"), LoadColumn(210)]
        public float D10r_d9r { get; set; }


        [ColumnName("d11r-d10r"), LoadColumn(211)]
        public float D11r_d10r { get; set; }


        [ColumnName("d12r-d11r"), LoadColumn(212)]
        public float D12r_d11r { get; set; }


        [ColumnName("dfr-d12r"), LoadColumn(213)]
        public float Dfr_d12r { get; set; }


        [ColumnName("alpha1"), LoadColumn(214)]
        public float Alpha1 { get; set; }


        [ColumnName("alpha2"), LoadColumn(215)]
        public float Alpha2 { get; set; }


        [ColumnName("alpha3"), LoadColumn(216)]
        public float Alpha3 { get; set; }


        [ColumnName("alpha4"), LoadColumn(217)]
        public float Alpha4 { get; set; }


        [ColumnName("alpha5"), LoadColumn(218)]
        public float Alpha5 { get; set; }


        [ColumnName("alpha6"), LoadColumn(219)]
        public float Alpha6 { get; set; }


        [ColumnName("alpha7"), LoadColumn(220)]
        public float Alpha7 { get; set; }


        [ColumnName("alpha8"), LoadColumn(221)]
        public float Alpha8 { get; set; }


        [ColumnName("alpha9"), LoadColumn(222)]
        public float Alpha9 { get; set; }


        [ColumnName("alpha10"), LoadColumn(223)]
        public float Alpha10 { get; set; }


        [ColumnName("alpha11"), LoadColumn(224)]
        public float Alpha11 { get; set; }


        [ColumnName("alpha12"), LoadColumn(225)]
        public float Alpha12 { get; set; }


        [ColumnName("alphaf"), LoadColumn(226)]
        public float Alphaf { get; set; }


        [ColumnName("alphan1"), LoadColumn(227)]
        public float Alphan1 { get; set; }


        [ColumnName("alphan2"), LoadColumn(228)]
        public float Alphan2 { get; set; }


        [ColumnName("alphan3"), LoadColumn(229)]
        public float Alphan3 { get; set; }


        [ColumnName("alphan4"), LoadColumn(230)]
        public float Alphan4 { get; set; }


        [ColumnName("alphan5"), LoadColumn(231)]
        public float Alphan5 { get; set; }


        [ColumnName("alphan6"), LoadColumn(232)]
        public float Alphan6 { get; set; }


        [ColumnName("alphan7"), LoadColumn(233)]
        public float Alphan7 { get; set; }


        [ColumnName("alphan8"), LoadColumn(234)]
        public float Alphan8 { get; set; }


        [ColumnName("alphan9"), LoadColumn(235)]
        public float Alphan9 { get; set; }


        [ColumnName("alphan10"), LoadColumn(236)]
        public float Alphan10 { get; set; }


        [ColumnName("alphan11"), LoadColumn(237)]
        public float Alphan11 { get; set; }


        [ColumnName("alphan12"), LoadColumn(238)]
        public float Alphan12 { get; set; }


        [ColumnName("alphanf"), LoadColumn(239)]
        public float Alphanf { get; set; }


        [ColumnName("beta1"), LoadColumn(240)]
        public float Beta1 { get; set; }


        [ColumnName("beta2"), LoadColumn(241)]
        public float Beta2 { get; set; }


        [ColumnName("beta3"), LoadColumn(242)]
        public float Beta3 { get; set; }


        [ColumnName("beta4"), LoadColumn(243)]
        public float Beta4 { get; set; }


        [ColumnName("beta5"), LoadColumn(244)]
        public float Beta5 { get; set; }


        [ColumnName("beta6"), LoadColumn(245)]
        public float Beta6 { get; set; }


        [ColumnName("beta7"), LoadColumn(246)]
        public float Beta7 { get; set; }


        [ColumnName("beta8"), LoadColumn(247)]
        public float Beta8 { get; set; }


        [ColumnName("beta9"), LoadColumn(248)]
        public float Beta9 { get; set; }


        [ColumnName("beta10"), LoadColumn(249)]
        public float Beta10 { get; set; }


        [ColumnName("beta11"), LoadColumn(250)]
        public float Beta11 { get; set; }


        [ColumnName("beta12"), LoadColumn(251)]
        public float Beta12 { get; set; }


        [ColumnName("betaf"), LoadColumn(252)]
        public float Betaf { get; set; }


        [ColumnName("alpha1-beta1"), LoadColumn(253)]
        public float Alpha1_beta1 { get; set; }


        [ColumnName("alpha2-beta2"), LoadColumn(254)]
        public float Alpha2_beta2 { get; set; }


        [ColumnName("alpha3-beta3"), LoadColumn(255)]
        public float Alpha3_beta3 { get; set; }


        [ColumnName("alpha4-beta4"), LoadColumn(256)]
        public float Alpha4_beta4 { get; set; }


        [ColumnName("alpha5-beta5"), LoadColumn(257)]
        public float Alpha5_beta5 { get; set; }


        [ColumnName("alpha6-beta6"), LoadColumn(258)]
        public float Alpha6_beta6 { get; set; }


        [ColumnName("alpha7-beta7"), LoadColumn(259)]
        public float Alpha7_beta7 { get; set; }


        [ColumnName("alpha8-beta8"), LoadColumn(260)]
        public float Alpha8_beta8 { get; set; }


        [ColumnName("alpha9-beta9"), LoadColumn(261)]
        public float Alpha9_beta9 { get; set; }


        [ColumnName("alpha10-beta10"), LoadColumn(262)]
        public float Alpha10_beta10 { get; set; }


        [ColumnName("alpha11-beta11"), LoadColumn(263)]
        public float Alpha11_beta11 { get; set; }


        [ColumnName("alpha12-beta12"), LoadColumn(264)]
        public float Alpha12_beta12 { get; set; }


        [ColumnName("alphaf-betaf"), LoadColumn(265)]
        public float Alphaf_betaf { get; set; }


        [ColumnName("type"), LoadColumn(266)]
        public string Type { get; set; }


        [ColumnName("score_val"), LoadColumn(267)]
        public float Score_val { get; set; }


        [ColumnName("Label"), LoadColumn(268)]
        public bool Label { get; set; }


        [ColumnName("Probability"), LoadColumn(268)]
        public float[] Probability { get; set; }

    }
}
