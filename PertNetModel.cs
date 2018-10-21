
﻿// local change 

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;
using MicrosoftResearch.Infer.Factors;
//comment on server 
namespace InferGeneNet
{
    public class PertNetModel
    {
        public InferenceEngine InferenceEngine;


        protected VariableArray<double> genesT1;
        protected VariableArray<double> genesT2;
        protected VariableArray<double> alpha;
        protected VariableArray<double> beta;
        public VariableArray<VariableArray<double>, double[][]> w;
        public Variable<int> NumGenes;

        protected VariableArray<Gaussian> genesT1Prior;
        //protected Variable<Gaussian> genesT2Prior;
        protected VariableArray<Gaussian> alphaPrior;
        protected VariableArray<Gaussian> betaPrior;
        protected VariableArray<VariableArray<Gaussian>, Gaussian[][]> wPrior;


        public virtual void CreateModel(int numGenes)
        {
            NumGenes = Variable.New<int>();
            NumGenes.ObservedValue = numGenes;

            Range geneRange = new Range(NumGenes).Named("geneRange");
            Range geneWeightRange = new Range(NumGenes - 1).Named("geneWeightRange");

            genesT1Prior = Variable.Array<Gaussian>(geneRange).Named("genesT1Prior"); ;
            alphaPrior = Variable.Array<Gaussian>(geneRange).Named("alphaPrior"); ;
            betaPrior = Variable.Array<Gaussian>(geneRange).Named("betaPrior");
            wPrior = Variable.Array(Variable.Array<Gaussian>(geneWeightRange), geneRange).Named("wPrior");

            genesT1 = Variable.Array<double>(geneRange).Named("genesT1");
            genesT2 = Variable.Array<double>(geneRange).Named("genesT2");

            var indicesArray = Variable.Array(Variable.Array<int>(geneWeightRange), geneRange).Named("indicesArray");

            //genesT1[geneRange] = Variable.Random<double, Gaussian>(genesT1Prior).ForEach(geneRange);
            genesT1[geneRange] = Variable<double>.Random(genesT1Prior[geneRange]);

            int[][] indices = new int[NumGenes.ObservedValue][];
            for (int i = 0; i < NumGenes.ObservedValue; i++)
            {
                indices[i] = new int[NumGenes.ObservedValue - 1];
                int j = 0, ind = 0;
                while (j < NumGenes.ObservedValue)
                {
                    if (i == j) { j++; continue; }
                    indices[i][ind] = j;
                    j++;
                    ind++;
                }
            }
            for (int i = 0; i < NumGenes.ObservedValue; i++) for (int j = 0; j < NumGenes.ObservedValue - 1; j++) { Console.WriteLine(indices[i][j]); }
            indicesArray.ObservedValue = indices;

            alpha = Variable.Array<double>(geneRange).Named("alpha");
            beta = Variable.Array<double>(geneRange).Named("beta");
            //alpha[geneRange] = Variable.Random<double, Gaussian>(alphaPrior).ForEach(geneRange);
            alpha[geneRange] = Variable<double>.Random(alphaPrior[geneRange]);
            //beta[geneRange] = Variable.Random<double, Gaussian>(betaPrior).ForEach(geneRange);
            beta[geneRange] = Variable<double>.Random(betaPrior[geneRange]);

            w = Variable.Array(Variable.Array<double>(geneWeightRange), geneRange).Named("w");
            w[geneRange][geneWeightRange] = Variable.Random<double, Gaussian>(wPrior[geneRange][geneWeightRange]); // Laplace distribution
            //w[geneRange][geneWeightRange] = Variable.GaussianFromMeanAndVariance(0, Variable.GammaFromShapeAndRate(1, 1)).ForEach(geneRange, geneWeightRange);


            VariableArray<double> genesubarray = Variable.Array<double>(geneWeightRange).Named("genesubarray");
            using (ForEachBlock firstBlock = Variable.ForEach(geneRange))
            {
                Variable<double> selfEffect = -(alpha[geneRange] * genesT1[geneRange]);
                VariableArray<double> weightSum = Variable.Array<double>(geneWeightRange).Named("weightSum");
                Console.WriteLine("summing " + geneRange + " + " + geneWeightRange);
                genesubarray = Variable.Subarray(genesT1, indicesArray[geneRange]);
                weightSum[geneWeightRange] = w[geneRange][geneWeightRange] * genesubarray[geneWeightRange];//genesT1[geneRange2]
                Variable<double> othersEffect = beta[geneRange] * Variable.Sum(weightSum);// Variable.Logistic(Variable.Sum(weightSum));// Variable.Sum(weightSum); //Variable.Logistic(innerproduct);
                genesT2[geneRange] = selfEffect + othersEffect;
            }

            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public virtual void SetModelData(NetModelData priors)
        {   
            genesT1Prior.ObservedValue = priors.genesT1Dist;
            alphaPrior.ObservedValue = priors.alphaDist;
            betaPrior.ObservedValue = priors.betaDist;
            wPrior.ObservedValue = priors.wDist;
        }

        public virtual void SetModelDataInit(NetModelData priors)
        {
            genesT1Prior.ObservedValue = priors.genesT1Dist;
            alphaPrior.ObservedValue = priors.alphaDist;
            betaPrior.ObservedValue = priors.betaDist;
            //wPrior.ObservedValue = priors.wDist;
            Range geneRange = new Range(priors.wDist.Length);
            Range geneWeightRange = new Range(priors.wDist.Length-1);
            // w[geneRange][geneWeightRange] = Variable.GaussianFromMeanAndVariance(0, Variable.GammaFromShapeAndRate(1, 1)).ForEach(geneRange, geneWeightRange);
            //wPrior.ObservedValue = Variable.GaussianFromMeanAndVariance(0, Variable.GammaFromShapeAndRate(1, 1)).ForEach(geneRange, geneWeightRange);
        }

        public NetModelData InferModelData(double[][] trainingData)
        {
            NetModelData posteriors;

            // NumGenes.ObservedValue = trainingData[0].Length;
            genesT1.ObservedValue = trainingData[0];
            genesT2.ObservedValue = trainingData[1];

            Console.WriteLine("\n\n** w: \n"+InferenceEngine.Infer(w));
            posteriors.wDist = InferenceEngine.Infer<Gaussian[][]>(w);
            posteriors.alphaDist = InferenceEngine.Infer<Gaussian[]>(alpha);
            posteriors.betaDist = InferenceEngine.Infer<Gaussian[]>(beta);
            posteriors.genesT1Dist = InferenceEngine.Infer<Gaussian[]>(genesT1);
            //posteriors.genesT2Dist = InferenceEngine.Infer<Gaussian>(genesT2);

            return posteriors;
        }

    }

    public struct NetModelData
    {
        public Gaussian[] genesT1Dist;
        public Gaussian[] alphaDist;
        public Gaussian[] betaDist;
        public Gaussian[][] wDist;

        public NetModelData(Gaussian[] genesT1, Gaussian[] alpha, Gaussian[] beta, Gaussian[][] w)
        {
            genesT1Dist = genesT1;
            //genesT2Dist = genesT2;
            alphaDist = alpha;
            betaDist = beta;
            wDist = w;
        }
    }
}
