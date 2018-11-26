using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Factors;
using InferGeneNet;
using MicrosoftResearch.Infer.Utils;

namespace InferGeneNet
{
    class Program
    {
        static void Main(string[] args)
        {

            double[][] trainingData = new double[2][];
            trainingData[0] = new double[] { 0.5, 0.6, 0.7, 0.24 };
            trainingData[1] = new double[] { 0.4, 0.3, 0.55, 0.95 };

            Gaussian alphaPrior = Gaussian.FromMeanAndVariance(1, 0.002);
            Gaussian betaPrior = Gaussian.FromMeanAndVariance(1, 0.002);
            Gaussian genesT1Prior = Gaussian.FromMeanAndVariance(1, 0.002);
            Gaussian wPrior = Gaussian.FromMeanAndVariance(0, 1);

            int nGenes = trainingData[0].Length;

            NetModelData initPriors = new NetModelData(
                     Util.ArrayInit(nGenes, u => genesT1Prior),
                     Util.ArrayInit(nGenes, u => alphaPrior),
                     Util.ArrayInit(nGenes, u => betaPrior),
                     Util.ArrayInit(nGenes, u => Util.ArrayInit(nGenes - 1, t => wPrior))
                ); // w   -> variance : Variable.GammaFromShapeAndRate(1, 1)
            //Train the model
            PertNetModel pertNetModel = new PertNetModel();
            Console.WriteLine("number of genes: " + trainingData[0].Length);
            pertNetModel.CreateModel(trainingData[0].Length);
            pertNetModel.SetModelData(initPriors);


            NetModelData posteriors1 = pertNetModel.InferModelData(trainingData);
            Gaussian[][] x = posteriors1.wDist;
            for (int i = 0; i < posteriors1.wDist.Length; i++ ) {
                for (int j = 0; j < posteriors1.wDist[i].Length; j++ ) {
                    Console.WriteLine(posteriors1.wDist[i][j]);
                }
            }
            //Console.WriteLine("Inferred w = " + posteriors1.wDist);
               // Console.WriteLine("===================");
               // Console.WriteLine(pertNetModel.w);
            //////////////////////////
            double[][] trainingData2 = new double[2][];
            trainingData2[0] = new double[] { 0.25, 0.16, 0.73, 0.4 };
            trainingData2[1] = new double[] { 0.94, 0.43, 0.25, 0.65 };

           pertNetModel.SetModelData(posteriors1);
           NetModelData posteriors2 = pertNetModel.InferModelData(trainingData2);

            Console.ReadLine();
        }
    }
}
