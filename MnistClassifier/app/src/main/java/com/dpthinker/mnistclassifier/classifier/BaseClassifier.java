package com.dpthinker.mnistclassifier.classifier;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.dpthinker.mnistclassifier.model.BaseModelConfig;
import com.dpthinker.mnistclassifier.model.ModelConfigFactory;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Comparator;
import java.util.Map;
import java.util.PriorityQueue;

public abstract class BaseClassifier {
    private final static String TAG = "BaseClassifier";
    protected static final int RESULTS_TO_SHOW = 3;
    protected Interpreter mTFLite;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    protected ByteBuffer mImgData;

    private String mModelPath = "";

    private int mNumBytesPerChannel;

    private int mDimBatchSize;
    private int mDimPixelSize;

    private int mDimImgWidth;
    private int mDimImgHeight;

    private BaseModelConfig mModelConfig;

    protected PriorityQueue<Map.Entry<String, Float>> mSortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    private void initConfig(BaseModelConfig config) {
        this.mModelConfig = config;
        this.mNumBytesPerChannel = config.getNumBytesPerChannel();
        this.mDimBatchSize = config.getDimBatchSize();
        this.mDimPixelSize = config.getDimPixelSize();
        this.mDimImgWidth = config.getDimImgWeight();
        this.mDimImgHeight = config.getDimImgHeight();
        this.mModelPath = config.getModelName();
    }

    public BaseClassifier(String modelConfig, Activity activity) throws IOException {
        initConfig(ModelConfigFactory.getModelConfig(modelConfig));

        // init interpreter with config parameter
        mTFLite = new Interpreter(loadModelFile(activity));

        mImgData = ByteBuffer.allocateDirect(
                mNumBytesPerChannel * mDimBatchSize * mDimImgWidth * mDimImgHeight * mDimPixelSize);
        mImgData.order(ByteOrder.nativeOrder());
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(mModelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    protected void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImgData == null) {
            return;
        }
        mImgData.rewind();

        int[] intValues = new int[mDimImgWidth * mDimImgHeight];
        scaleBitmap(bitmap).getPixels(intValues,
                0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        long startTime = SystemClock.uptimeMillis();
        // Convert the image to floating point.
        int pixel = 0;
        for (int i = 0; i < mDimImgWidth; ++i) {
            for (int j = 0; j < mDimImgHeight; ++j) {
                final int val = intValues[pixel++];
                mModelConfig.addImgValue(mImgData, val);
            }

        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + (endTime - startTime));
    }

    public Bitmap scaleBitmap(Bitmap bmp) {
        return Bitmap.createScaledBitmap(bmp, mDimImgWidth, mDimImgHeight, true);
    }

    public abstract String printTopKLabels();

    public abstract String doClassify(Bitmap bmp);
}
