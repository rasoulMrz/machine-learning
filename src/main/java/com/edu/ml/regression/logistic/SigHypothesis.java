package com.edu.ml.regression.logistic;

import com.edu.ml.regression.RegressionHypothesis;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Sigmoid;
import org.nd4j.linalg.factory.Nd4j;

public class SigHypothesis extends RegressionHypothesis
{
    @Override
    public INDArray computeValue(INDArray theta, INDArray x)
    {
        INDArray v = x.mulRowVector(theta).sum(1);
        Nd4j.getExecutioner().execAndReturn(new Sigmoid(v));
        return v.reshape(v.columns(), v.rows());
    }
}
