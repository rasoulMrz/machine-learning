package com.edu.ml;

import java.util.function.Predicate;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class GradientDecent
{
    private final DataSet dataSet;
    private final INDArray x;
    private final Hypothesis hypothesis;
    private Double learningRate;
    private INDArray theta;
    private int iterationCount = 0;

    public GradientDecent(DataSet dataSet, Hypothesis hypothesis, Double learningRate)
    {
        this(dataSet, hypothesis, learningRate, Nd4j.rand(1, dataSet.getFeatures().columns() + 1));
    }

    public GradientDecent(DataSet dataSet, Hypothesis hypothesis, Double learningRate, INDArray theta)
    {
        this.dataSet = dataSet;
        this.theta = theta;
        this.hypothesis = hypothesis;
        this.learningRate = learningRate;
        x = Nd4j.hstack(Nd4j.ones(dataSet.getFeatures().rows(), 1), dataSet.getFeatures());
    }

    public Summary iterate()
    {
        INDArray hValues = hypothesis.computeValue(theta, x);
        INDArray grad = hypothesis.computeGradien(hValues, x, dataSet.getLabels());
        INDArray preTheta = theta;
        theta = (grad.muli(learningRate).negi()).add(theta);
        iterationCount++;
        return new Summary(preTheta, theta, hValues, iterationCount);
    }

    public void iterateWhile(Predicate<Summary> predicate)
    {
        while (predicate.test(iterate()));
    }

    public void run(CostFunction costFunction)
    {
        INDArray ones = Nd4j.ones(1, 1);
        iterateWhile(summary -> {
            if ((summary.iteration % 10) == 0) {
                System.out.println("--------------iteration----------------");
                System.out.println(summary.iteration);
                System.out.println(theta);
                double cost = costFunction.compute(summary.hValues, dataSet.getLabels());
                System.out.println(cost);
                double delta = ones.getDouble(0, 0) - cost;
                System.out.println(delta);
                if (delta < 0) {
                    System.out.println("Error increased");
//                    return false;
                    learningRate *= 0.7;
                }
                ones.getRow(0).putScalar(0, cost);
            }
            return !summary.preTheta.equalsWithEps(summary.theta, 0.0001);
        });
    }

    public Double getLearningRate()
    {
        return learningRate;
    }

    public void setLearningRate(Double learningRate)
    {
        this.learningRate = learningRate;
    }

    public INDArray getTheta()
    {
        return theta;
    }

    public void setTheta(INDArray theta)
    {
        this.theta = theta;
    }

    public static class Summary
    {
        private final INDArray preTheta;
        private final INDArray theta;
        private final INDArray hValues;
        private final long iteration;

        public Summary(INDArray preTheta, INDArray theta, INDArray hValues, long iteration)
        {
            this.preTheta = preTheta;
            this.theta = theta;
            this.hValues = hValues;
            this.iteration = iteration;
        }
    }

}
