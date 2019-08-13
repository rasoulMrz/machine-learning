/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.edu.ml.regression.logistic;

import java.io.IOException;

public class PolyData
{
    private static final String DATA_STORE = "poly";
    private static final String CSV = "poly.csv";

    public static void main(String[] args) throws IOException
    {
        LogisticRegEngine lre = new LogisticRegEngine(DATA_STORE, 2);
        lre.convertAndSaveData(CSV, 15, 14, 2);
        lre.train();
        lre.test();
    }

}
