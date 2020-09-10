using System;
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.ML;

namespace VIC
{
    class Clustering
    {
        private List<List<Object>> sorted_data;

        public Clustering(string data_path)
        {
            sorted_data = load_csv(data_path);
        }


        /*load a csv with the data into a List of Lists and sort it by the score_chage column*/
        private List<List<Object>> load_csv(string data_path)
        {
            System.IO.StreamReader file = new System.IO.StreamReader(data_path);

            string[] test = file.ReadLine().Split(',') ; //skipping header
            Debug.WriteLine(test[267]);

            string line;
            int line_count = 1;
            List<List<Object>> data = new List<List<Object>>();
            while ((line = file.ReadLine()) != null)
            {
                string [] str_data = line.Split(',');
                line_count += 1;
                if(str_data.Length == 269)
                {
                   List<Object> row = new List<object>(capacity: 269);
                   foreach(string element in str_data)
                    {
                        row.Add(fix_type(element));
                    }
                    row.Add(null);
                    data.Add(row);
                }
                else
                {
                    System.Console.WriteLine(line_count);
                }
                
            }
            file.Close();

            data.Sort(delegate(List<Object> row1, List<Object> row2) {
                float x = (float)row1[267];
                float y = (float)row2[267];
                if (x > y) return 1;
                else if (x == y) return 0;
                else return -1;
            });

            return data;
        }


        /*cast numeric attributes from string to float, keep strings as strings and set missing values
         * to a invalid number (-1).
         */
        private Object fix_type(string element)
        {
            float numeric_rv;
            if(float.TryParse(element, out numeric_rv))
            {
                return numeric_rv;
            }
            else
            {
                if (element == "")
                {
                    return -1.0f;
                }
                return element;
            }
        }

        //########################clustering#################################################

        /*create partitions of 2 clusters. The partition returned is determinated by "cluster"
         * 0<=cluster <50
        */
        public IDataView get2clustered(int cluster)
        {
            if (cluster >= 50)
            {
                System.Console.WriteLine("More than 50 calls (2-clustered)");
            }
            float clusters_split = (sorted_data.Count / 51.0f) * (cluster + 1);
            MinutiaData[] in_memory = new MinutiaData[sorted_data.Count];
            for (int i=0; i<sorted_data.Count;++i)
            {
                sorted_data[i][269] = (i < clusters_split)? 0:1;
                in_memory[i] = fromRow2Data(sorted_data[i]);
            }

            return new MLContext().Data.LoadFromEnumerable<MinutiaData>(in_memory);

        }

        /*
         * returns the number of divisions needed to obtain "c" diferenet partitions of 3 clusters
         */
        private int cnt_seg(int c)
        {
            return (int)Math.Ceiling((Math.Sqrt(1 + 8 * c) + 3) / 2.0);
        }

        /*
         * Find the limits of the first and second cluster for the "cluster^th" partition. (for 3 clusters partitions)
         */
        private int[] find_idx(int cluster, int segs)
        {
            int idx_1 = 0;
            int act_idx = -1;
            while (act_idx < cluster)
            {
                act_idx += segs - 1 - ++idx_1;

            }
            return new int[] { idx_1, segs - 1 - act_idx + cluster };
        }


        /*create partitions of 3 clusters. The partition returned is determinated by "cluster"
         * 0<=cluster <50
         */
        public IDataView get3clustered(int cluster)
        {
            if (cluster >= 50)
            {
                System.Console.WriteLine("More than 50 calls (3-clustered)");
            }

            float segs = cnt_seg(50);
            float seg_lenght = sorted_data.Count / segs;

            int[] idxs = find_idx(cluster, (int)segs);
            int clusters3idx_2 = idxs[1];
            int clusters3idx_1 = idxs[0];

            int end_1 = (int)(seg_lenght * clusters3idx_1);
            int end_2 = (int)(seg_lenght * clusters3idx_2);

            MinutiaData[] in_memory = new MinutiaData[sorted_data.Count];
            for (int i = 0; i < sorted_data.Count; ++i)
            {
                //labels for the clusters are are int (0,1,2)
                if (i < end_1)
                {
                    sorted_data[i][269] = 0;
                }
                else if (i < end_2)
                {
                    sorted_data[i][269] = 1;
                }
                else
                {
                    sorted_data[i][269] = 2;
                }
                in_memory[i] = fromRow2Data(sorted_data[i]);
            }
            return new MLContext().Data.LoadFromEnumerable<MinutiaData>(in_memory);
        }

        /*
         * tranforms a List with attributes of a minutia in a MinutiaData Model for ML.NET
         */
        private MinutiaData fromRow2Data(List<Object> row)
        {
            return new MinutiaData
            {
                //fingerprint = (string)row[0],
                //minutia = (string)row[1],
                nn = Array.ConvertAll(slice(row, 2, 41), item => (float)item),
                nnr = Array.ConvertAll(slice(row, 43, 41), item => (float)item),
                nn_nn = Array.ConvertAll(slice(row, 84, 40), item => (float)item),
                nnr_nnr = Array.ConvertAll(slice(row, 124, 40), item => (float)item),
                dn = Array.ConvertAll(slice(row, 164, 12), item => (float)item),
                df = (float)row[176],
                dnr = Array.ConvertAll(slice(row, 177, 12), item => (float)item),
                dfr = (float)row[189],
                dn_dn = Array.ConvertAll(slice(row, 190, 12), item => (float)item),
                dnr_dnr = Array.ConvertAll(slice(row, 202, 12), item => (float)item),
                alphan = Array.ConvertAll(slice(row, 214, 12), item => (float)item),
                alphaf = (float)row[226],
                alphann = Array.ConvertAll(slice(row, 227, 12), item => (float)item),
                alphanf = (float)row[239],
                betann = Array.ConvertAll(slice(row, 240, 12), item => (float)item),
                betaf = (float)row[252],
                alphan_betan = Array.ConvertAll(slice(row, 253, 13), item => (float)item),
                type = (string)row[266],
                score = (float)row[267],
                //discretized_class = (float)row[268],
                Label =  Convert.ToInt32(row[269])
            };
        }

        /*
         * return a slice of a list in the form of a Object array
         * starting in "start" (inclusive) and taking "amount" consecutive elements
         */
        private Object[] slice(List<Object> row, int start, int amount)
        {
            Object[] rv = new Object[amount];
            for (int i = 0; i < amount; ++i)
            {
                rv[i] = row[start + i];
            }
            return rv;
        }
    }
}
