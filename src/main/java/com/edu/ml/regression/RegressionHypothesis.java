package com.edu.ml.regression;

import com.edu.ml.Hypothesis;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class RegressionHypothesis implements Hypothesis
{
    @Override
    public INDArray computeGradien(INDArray value, INDArray x, INDArray labels)
    {
        return x.mulColumnVector(labels.neg().addi(value)).sum(0);
    }
}
