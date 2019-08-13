/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.edu.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Hypothesis
{
    INDArray computeValue(INDArray theta, INDArray x);

    INDArray computeGradien(INDArray value, INDArray x, INDArray labels);
}
