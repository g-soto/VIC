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


        /*multi-class AUC from https://github.com/miguelmedinaperez/DTAE/blob/master/core/AUCCalculatorExtensions.cs*/
        /****************************BEGINNING***********************************************/
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
        /******************************************************END********************************************/

        /*compute the AUC of model in the trainingData2 with 10-folds cross validation.
         * Number of clases in trainingData2 must be indicated in classNumber*/

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

        /*VIC algorithm. 
         * arguments
         * get_cluster: a function that recive an int and return a corresponding partition in a IDataView form.
         * number_of_clusters: number of clusters in the partitions.
         * number_of_partitions: number of partitions that will be generated using get_cluster.
         * get_model: a function that recive an int and a MLContext, and return a classificator in the MLContext.
         * number_of_models: number of models that will be generated using get_model.
         * mlContext: MLContext that will be used in the get_model function.
         * core: number of parallel proccess for the VIC algorithm.
         * 
         * return
         * bi-dimensional double array with number_of_partitions rows and number_of_models columns, with the AUC of each model for each partition.
         * */
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

                aucArray[partition_number, model_number] = CalculateModelAUC(mlContext, get_model(model_number, mlContext), partitions_cache[partition_number], number_of_clusters);
            });


            return aucArray;
        }


        /*function for the VIC algorithm
         * return an element a model in "mlContext" depending of the "idx"
         */
        public static IEstimator<ITransformer> get_model(int idx, MLContext mlContext)
        {
            switch (idx)
            {
                case 0:
                    return mlContext.MulticlassClassification.Trainers.NaiveBayes();
                case 1:
                    return mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastTree());
                case 2:
                    return mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.FastForest());
                case 3:
                    return mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());
                case 4:
                    return mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron());
                case 5:
                    return mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LinearSvm());
                case 6:
                    return mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy();
                default:
                    return null;
            }
        }

        /*find the row wise maximun elements of a bidimensional array*/
        public static double[] max_axis1(double[,] arr)
        {
            double[] rv = new double[arr.GetUpperBound(0)+1];
            for (int p = 0; p<=arr.GetUpperBound(0); ++p)
            {
                rv[p] = arr[p, 0];
                for (int m = 1; m<=arr.GetUpperBound(1); ++m)
                {
                    if (arr[p,m] > rv[p])
                    {
                        rv[p] = arr[p, m];
                    }
                }
            }

            return rv;
        }

        /*Write in the standard output a double array*/
        public static void print(double[] arr)
        {
            string line = "";
            for(int k =0; k<arr.Length; ++k)
            {
                line += "," + arr[k];
            }
            System.Console.WriteLine(line.Substring(1));
        }


        /*program's entry point. use VIC algorithm in partitions of 2 and 3 clusters. 
         * Write the AUC of each classificator for each partitions to a csv file.
         * Write to the standard output the VIC for each partition
         */
        static void Main(string[] args)
        {

            MLContext mlContext = new MLContext(seed: 0);

            Clustering clustering = new Clustering(@"D:\Code\Migue\Assignment_2\VIC_classifiers\VIC\data2.csv");

            double[,] AUCs2 = vic(clustering.get2clustered, 2, 50, get_model, 7, mlContext, 6);

            double[,] AUCs3 = vic(clustering.get3clustered, 3, 50, get_model, 7, mlContext, 6);

            Util.save2csv("2c.csv", AUCs2);

            Util.save2csv("3c.csv", AUCs3);

            double[] vic2 = max_axis1(AUCs2);

            double[] vic3 = max_axis1(AUCs3);

            System.Console.WriteLine("VIC for 2-clusters partitions");
            print(vic2);

            System.Console.WriteLine("VIC for 3-clusters partitions");
            print(vic3);
        }


    }
}