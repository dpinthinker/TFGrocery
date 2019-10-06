package com.dpthinker.mnistclassifier.classifier;

import android.app.Activity;
import android.graphics.Bitmap;
import android.util.Log;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.Map;

public class FloatClassifier extends BaseClassifier {
    private static final String TAG = "FloatClassifier";
    private float[][] mLabelProbArray = new float[1][10];
    //private byte[][] mLabelProbArray = new byte[1][10];

    public FloatClassifier(String modelConfig, Activity activity) throws IOException {
        super(modelConfig, activity);
    }

    @Override
    public String doClassify(Bitmap bitmap) {
        convertBitmapToByteBuffer(bitmap);
        mTFLite.run(mImgData, mLabelProbArray);
        String result = printTopKLabels();
        return result;
    }

    /** Prints top-K labels, to be shown in UI as the results. */
    public String printTopKLabels() {
        for (int i = 0; i < 10; ++i) {
            mSortedLabels.add(new AbstractMap.SimpleEntry<>(""+i, mLabelProbArray[0][i]));
            //mSortedLabels.add(new AbstractMap.SimpleEntry<>(""+i, (mLabelProbArray[0][i]& 0xff) / 255.0f));
            if (mSortedLabels.size() > RESULTS_TO_SHOW) {
                mSortedLabels.poll();
            }
        }
        StringBuffer textToShow = new StringBuffer();
        final int size = mSortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = mSortedLabels.poll();
            textToShow.insert(0, String.format("\n%s   %4.8f",label.getKey(),label.getValue()));
        }
        return textToShow.toString();
    }
}
