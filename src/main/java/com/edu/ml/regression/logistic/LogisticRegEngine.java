package com.edu.ml.regression.logistic;

import com.edu.ml.GradientDecent;
import com.edu.ml.regression.LinearHypothesis;
import java.io.File;
import java.io.IOException;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

public class LogisticRegEngine
{
    private static final String TRAIN_DATA = "trainDatSet";
    private static final String TEST_DATA = "testDataSet";
    private static final String NORMALIZER = "normalizer";
    private static final String THETA = "theta";

    private final String dataDropDir;
    private final int degree;

    public LogisticRegEngine(String dataDropDir, int degree)
    {
        this.dataDropDir = dataDropDir;
        this.degree = degree;
    }

    public void train()
    {
        try {
            DataSet train = new DataSet();
            train.load(new File(filePath(TRAIN_DATA)));
            prepareFeatures(train);
            MeanNormalizer mn = new MeanNormalizer();
            mn.normalizeTrainingSet(train);
            mn.saveParams(filePath(NORMALIZER));
            GradientDecent grad = new GradientDecent(train, new SigHypothesis(), 0.001);
            grad.run(new LogisticCostFunction());
            INDArray theta = grad.getTheta();
            Nd4j.saveBinary(theta, new File(filePath(THETA)));
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    public void test()
    {
        try {
            INDArray theta = Nd4j.readBinary(new File(filePath(THETA)));
            DataSet test = new DataSet();
            test.load(new File(filePath(TEST_DATA)));
            prepareFeatures(test);
            MeanNormalizer mn = new MeanNormalizer();
            mn.loadParams(filePath(NORMALIZER));
            mn.normalizeTestSet(test);
            INDArray features = test.getFeatures();
            LinearHypothesis h = new LinearHypothesis();
            INDArray x = Nd4j.hstack(Nd4j.ones(features.rows(), 1), features);
            INDArray val = h.computeValue(theta, x);
            int err = 0;
            for (int i = 0; i < features.rows(); i++) {
                double pred = (val.getDouble(i) <= 0) ? 0 : 1;
                if (pred != test.getLabels().getDouble(i, 1))
                    err++;
            }
            System.out.println("Error");
            System.out.println(err);
            System.out.println(err * 100d / features.rows());
        } catch (IOException ex) {
            throw new RuntimeException(ex);
        }
    }

    protected void prepareFeatures(DataSet ds)
    {
        int origFeatures = ds.getFeatures().columns();
        INDArray features = ds.getFeatures();
        for (int j = 1; j < degree; j++) {
            INDArray f = features;
            for (int i = 0; i < origFeatures; i++) {
//                INDArray col = features.getColumn(i).reshape(features.rows(), 1);
                INDArray col = features.getColumn(i);
//                INDArray mmm = col.mulColumnVector(col);
                for (int k = i; k < features.columns(); k++) {
                    INDArray col2 = features.getColumn(k);
                    f = Nd4j.hstack(f, col.mulRowVector(col2).reshape(features.rows(), 1));
                }
            }
            features = f;
        }
        ds.setFeatures(features);
    }

    public void convertAndSaveData(String dataSheet, int trainSize, int testSize, int labelIndex)
    {
        File file = new File(filePath(dataSheet));
        convertToDataSet(file, 1, trainSize, labelIndex).save(new File(filePath(TRAIN_DATA)));
        convertToDataSet(file, trainSize + 1, testSize, labelIndex).save(new File(filePath(TEST_DATA)));
    }

    private String filePath(String name)
    {
        return dataDropDir + File.separator + name;
    }

    private static DataSet convertToDataSet(File file, int from, int size, int labelIndex)
    {
        RecordReader rr = new CSVRecordReader(from);
        try {
            rr.initialize(new FileSplit(file));
            DataSetIterator iter = new RecordReaderDataSetIterator(rr, size, labelIndex, 1);
            DataSet ds = iter.next();
            return ds;
        } catch (IOException | InterruptedException ex) {
            throw new RuntimeException(ex);
        }
    }
}
