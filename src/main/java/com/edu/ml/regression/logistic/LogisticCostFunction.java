package com.edu.ml.regression.logistic;

import com.edu.ml.CostFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.strict.Log;
import org.nd4j.linalg.factory.Nd4j;

public class LogisticCostFunction implements CostFunction
{
    @Override
    public double compute(INDArray hValue, INDArray labels)
    {
        INDArray v1 = hValue.transpose().dup();
        INDArray v2 = v1.neg().addi(1);
        Nd4j.getExecutioner().execAndReturn(new Log(v1));
        Nd4j.getExecutioner().execAndReturn(new Log(v2));
        return -1 * (Nd4j.matmul(v1, labels).getDouble(0)
                + Nd4j.matmul(v2, labels.neg().addi(1)).getDouble(0)) / labels.rows();
    }

}
