/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.edu.ml;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface CostFunction
{
    double compute(INDArray hValue, INDArray labels);
}
