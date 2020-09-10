﻿using Microsoft.ML;
using System;
using Microsoft.ML.Data;
using System.Linq;
using System.Threading.Tasks;

namespace VIC
{
    class VIC
    {

        public class ModelOutput
        {
            // ColumnName attribute is used to change the column name from
            // its default value, which is the name of the field.
            [ColumnName("PredictedLabel")]
            public String Prediction { get; set; }
            public float[] Score { get; set; }
            public float[] Probability { get; set; }
        }

        public class Prediction
        {
            [ColumnName("Score")]
            public float Price { get; set; }
        }


        public class BasicEvaluation
        {
            public int TP = 0;
            public int TN = 0;
            public int FN = 0;
            public int FP = 0;
        }

        public static double ComputeTwoClassAUC(BasicEvaluation basicEvaluation)
        {
            double positives = basicEvaluation.TP + basicEvaluation.FN;
            double negatives = basicEvaluation.TN + basicEvaluation.FP;
            var tprate = positives > 0.0 ? basicEvaluation.TP / positives : 1.0;
            var fprate = negatives > 0.0 ? basicEvaluation.TN / negatives : 1.0;
            return (tprate + fprate) / 2.0;
        }

        public static double ComputeMultiClassAUC(int[,] confusionMatrix)
        {
            var eval = new BasicEvaluation();
            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                eval.TP += confusionMatrix[i, i];
                for (int j = 0; j < confusionMatrix.GetLength(0); j++)
                    if (i != j)
                    {
                        eval.FN += confusionMatrix[i, j];
                        eval.FP += confusionMatrix[j, i];

                        eval.TN += confusionMatrix[j, j];
                    }

            }
            return ComputeTwoClassAUC(eval);
        }

        public static double CalculateModelAUC(MLContext mlContext, IEstimator<ITransformer> model, IDataView trainingData2, int classNumber)
        {
            string[] attributes = {
                "nn",
                "nnr"   ,
                "nn_nn"   ,
                "nnr_nnr" ,
                "dn"  ,
                "df"  ,
                "dnr" ,
                "dfr" ,
                "dn_dn"   ,
                "dnr_dnr" ,
                "alphan"  ,
                "alphaf"  ,
                "alphann" ,
                "alphanf" ,
                "betann"   ,
                "betaf"   ,
                "alphan_betan"
                };

            // 3. Model selection and confusion matrix creation, the matrix is not filled at this time.
            int[,] matriz = new int[classNumber, classNumber];

            double aucValues = 0;

            // 4. Pipeline creation, cross validation evaluation and AUC computation through the confusion matrix 
            var pipeline0 = mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "typeEncoded", inputColumnName: "type").Append(mlContext.Transforms.Concatenate("Features", attributes))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label")).Append(model); //pipeline for specific model
            var scores0 = mlContext.MulticlassClassification.CrossValidate(trainingData2, pipeline0, numberOfFolds: 10); //cross validation evaluation.
            aucValues = 0;
            for (int k = 0; k < 10; k++) //confusion matrix generation
            {
                for (int i = 0; i < classNumber; i++)
                {
                    for (int j = 0; j < classNumber; j++)
                    {
                        matriz[i, j] = (int)scores0[k].Metrics.ConfusionMatrix.GetCountForClassPair(i, j);
                    }
                }
                aucValues += ComputeMultiClassAUC(matriz); //computes multi class auc and adds it to be calculated as an average. 
            }

            return aucValues / 10; //return mean AUC 
        }

        static double[,] vic(Func<int, IDataView> get_cluster, int number_of_clusters, int number_of_partitions, Func<int, MLContext, IEstimator<ITransformer>> get_model, int number_of_models, MLContext mlContext, int cores = 7)
        {

            double[,] aucArray = new double[number_of_partitions, number_of_models];
            int[] indexArray = new int[number_of_models * number_of_partitions];

            for (int i = 0; i < indexArray.Length; ++i)
            {
                indexArray[i] = i;
            }

            IDataView[] partitions_cache = new IDataView[number_of_partitions];

            Parallel.ForEach(indexArray, new ParallelOptions { MaxDegreeOfParallelism = cores }, (x) =>
            {
                int model_number = x % number_of_models;
                int partition_number = x / number_of_models;
                System.Console.WriteLine("model {0} partition{1}", model_number, partition_number);
                lock ("cache")
                {
                    if (partitions_cache[partition_number] is null)
                    {
                        partitions_cache[partition_number] = get_cluster(partition_number);
                    }
                }

                aucArray[partition_number, model_number] = CalculateModelAUC(mlContext, get_model(model_number, mlContext), partitions_cache[partition_number], 2);
            });


            return aucArray;
        }

        public static IEstimator<ITransformer> get_model(int idx, MLContext mlContext)
        {
            IEstimator<ITransformer>[] models = { mlContext.MulticlassClassification.Trainers.NaiveBayes(),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron()),
                mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LinearSvm()),
                mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy() };
            return models[idx];
        }

        static void Main(string[] args)
        {

            MLContext mlContext = new MLContext(seed: 0);

            Clustering clustering = new Clustering();

            Util.save2csv("2c.csv", vic(clustering.get2clustered, 2, 50, get_model, 7, mlContext, 6));

            Util.save2csv("3c.csv", vic(clustering.get3clustered, 3, 50, get_model, 7, mlContext, 6));

        }


    }
}