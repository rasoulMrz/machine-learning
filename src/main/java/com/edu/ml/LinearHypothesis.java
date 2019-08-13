/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.edu.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public class LinearHypothesis extends RegressionHypothesis
{
    @Override
    public INDArray computeValue(INDArray theta, INDArray x)
    {
        INDArray v = x.mulRowVector(theta).sum(1);
        return v.reshape(v.columns(), v.rows());
    }
}
