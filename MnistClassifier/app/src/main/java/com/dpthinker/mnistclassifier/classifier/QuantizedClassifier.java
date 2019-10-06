package com.dpthinker.mnistclassifier.classifier;

import android.app.Activity;
import android.graphics.Bitmap;

import java.io.IOException;

public class QuantizedClassifier extends BaseClassifier {
    public QuantizedClassifier(String modelConfig, Activity activity) throws IOException {
        super(modelConfig, activity);
    }

    @Override
    public String printTopKLabels() {
        return null;
    }

    @Override
    public String doClassify(Bitmap bmp) {
        return null;
    }
}
