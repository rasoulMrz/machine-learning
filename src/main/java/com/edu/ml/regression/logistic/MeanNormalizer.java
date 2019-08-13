/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.edu.ml.regression.logistic;

import com.google.common.base.Preconditions;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class MeanNormalizer
{
    private INDArray params;//[[avg, range], ...]

    public void normalizeTrainingSet(DataSet ds)
    {
        Preconditions.checkState(params == null);
        INDArray features = ds.getFeatures();
        fixTrainMissings(features);
        computeParams(features);
        normalize(features);
    }

    public void normalizeTestSet(DataSet ds)
    {
        Preconditions.checkState(params != null);
        INDArray features = ds.getFeatures();
        fixTestMissings(features);
        normalize(features);
    }

    public void loadParams(String fileName)
    {
        try {
            params = Nd4j.read(new FileInputStream(fileName));
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    public void saveParams(String fileName)
    {
        Preconditions.checkState(params != null);
        try {
            Nd4j.write(new FileOutputStream(fileName), params);
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    private void normalize(INDArray features)
    {
        features.addiRowVector(params.getRow(0).neg());
        features.diviRowVector(params.getRow(1));
    }

    private void fixTrainMissings(INDArray features)
    {
        Preconditions.checkState(params == null);
        for (int i = 0; i < features.columns(); i++) {
            INDArray column = features.getColumn(i);
            List<Integer> missings = new ArrayList<>();
            double sum = 0;
            for (int j = 0; j < column.columns(); j++) {
                double d = column.getDouble(j);
                if (Double.isNaN(d)) {
                    missings.add(j);
                } else sum += d;
            }
            double avg = sum / (column.columns() - missings.size());
            for (Integer idx : missings)
                column.putScalar(idx, avg);
        }
    }

    private void fixTestMissings(INDArray features)
    {
        Preconditions.checkState(params != null);
        for (int i = 0; i < features.columns(); i++) {
            INDArray column = features.getColumn(i);
            double avg = params.getDouble(0, i);
            for (int j = 0; j < column.columns(); j++) {
                double d = column.getDouble(j);
                if (Double.isNaN(d))
                    column.putScalar(j, avg);
            }
        }
    }

    private void computeParams(INDArray features)
    {
        params = Nd4j.concat(0, features.mean(0), features.max(0).add(features.min(0).negi())).reshape(2, features.
                columns());
    }

}
