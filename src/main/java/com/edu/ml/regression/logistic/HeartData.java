/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.edu.ml.regression.logistic;

import java.io.IOException;

public class HeartData
{
    private static final String DATA_STORE = "datastore";
    private static final String CSV = "framingham.csv";

    public static void main(String[] args) throws IOException
    {
        LogisticRegEngine lre = new LogisticRegEngine(DATA_STORE, 2);
        lre.convertAndSaveData(CSV, 3000, 1200, 15);
        lre.train();
        lre.test();
    }

}
