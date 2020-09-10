using System.IO;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;


namespace VIC
{
    class Util
    {
        /*convert from arff to csv, the output is writed in the same folder than the input file
         * only difference between both files is the extension
         */
        public static void Arff2csv(string arff_path = @"D:\Code\Migue\Assignment_2\VIC\VIC\data.arff")
        {
            string[] text = System.IO.File.ReadAllText(arff_path).Split("@data"); 

            //@attribute to column name
            Regex rx = new Regex(@"@attribute (?<attr_name>[\w-]+|('[\w-]+( [\w-]+)+')) .+", RegexOptions.Compiled);
            string header = "";
            foreach (Match m in rx.Matches(text[0]))
            {
                header += ',' + m.Groups["attr_name"].Value;
            }

            //arff missing values (?) are replaced for csv missing values (",")
            text[1] = text[1].Replace("?","");

            string output_path = Path.Join(Path.GetDirectoryName(arff_path), Path.GetFileNameWithoutExtension(arff_path) + ".csv");
            using (System.IO.StreamWriter file = new System.IO.StreamWriter(output_path))
            {
                file.WriteLine(header.Substring(1));
                file.Write(text[1]);
            }
        }


        /*save a bidemensional double array to a csv file*/

        public static void save2csv(string filename, double[,] arr)
        {

            using (StreamWriter writer = new StreamWriter(filename, false, System.Text.Encoding.UTF8))
            {
                for (int p = 0; p <= arr.GetUpperBound(0); ++p)
                {
                    string line = "";
                    for (int m = 0; m <= arr.GetUpperBound(1); ++m)
                    {
                        line += "," + arr[p, m];
                    }
                    writer.WriteLine(line.Substring(1));
                }

            }
        }
    }
}
