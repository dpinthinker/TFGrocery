package com.dpthinker.mnistclassifier;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import com.dpthinker.mnistclassifier.model.BaseModelConfig;
import com.dpthinker.mnistclassifier.model.ModelConfigFactory;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.Comparator;
import java.util.Map;
import java.util.PriorityQueue;

public class MainActivity extends AppCompatActivity {

    private final static String TAG = "MainActivity";
    private String mModelPath = "inceptionv3_mnist.tflite";

    private int mNumBytesPerChannel = 4;

    private int mDimBatchSize = 1;
    private int mDimPixelSize = 3;

    private int mDimImgWidth = 299;
    private int mDimImgHeight = 299;

    private static final int RESULTS_TO_SHOW = 3;

    private int mImageMean = 0;
    private float mImageSTD = 255.0f;

    private Interpreter mTFLite = null;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private ByteBuffer mImgData = null;

    /** Preallocated buffers for storing image data in. */
    private int[] mIntValues = new int[mDimImgWidth * mDimImgHeight];

    //private float[][] mLabelProbArray = new float[1][10];
    private byte[][] mLabelProbArray = new byte[1][10];

    private BaseModelConfig mModelConfig = null;

    private PriorityQueue<Map.Entry<String, Float>> mSortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(mModelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void initConfig(BaseModelConfig config) {
        this.mNumBytesPerChannel = config.getNumBytesPerChannel();
        this.mDimBatchSize = config.getDimBatchSize();
        this.mDimPixelSize = config.getDimPixelSize();
        this.mDimImgWidth = config.getDimImgWeight();
        this.mDimImgHeight = config.getDimImgHeight();
        this.mImageMean = config.getImageMean();
        this.mImageSTD = config.getImageSTD();
        this.mModelPath = config.getModelName();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mModelConfig = ModelConfigFactory.getModelConfig(
                ModelConfigFactory.INCEPTION_QUANT_MODEL);
        initConfig(mModelConfig);
        try {
            mTFLite = new Interpreter(loadModelFile(this));
            mImgData = ByteBuffer.allocateDirect(
                    mNumBytesPerChannel * mDimBatchSize * mDimImgWidth * mDimImgHeight * mDimPixelSize);
            mImgData.order(ByteOrder.nativeOrder());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        Bitmap bmp = getBitmapFromAssets("s1.jpg");
        convertBitmapToByteBuffer(bmp);
        Drawable drawable = new BitmapDrawable(getResources(), bmp);
        ImageView imageView = findViewById(R.id.mnist_img);
        imageView.setImageDrawable(drawable);

        mTFLite.run(mImgData, mLabelProbArray);
        Log.e(TAG, "result size: " + mLabelProbArray[0].length);
        for (int i = 0; i < 10; i++) {
            Log.e(TAG, "index " + i + " prob is " + (mLabelProbArray[0][i]& 0xff) / 255.0f);
        }

        TextView textView = findViewById(R.id.tv_prob);
        textView.setText(printTopKLabels());
    }

    public Bitmap getBitmapFromAssets(String fileName) {
        AssetManager assetManager = getAssets();

        InputStream inputStream = null;
        try {
            inputStream = assetManager.open(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

        bitmap = Bitmap.createScaledBitmap(bitmap, mDimImgWidth, mDimImgHeight, true);

        return bitmap;
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (mImgData == null) {
            return;
        }
        mImgData.rewind();
        Log.d(TAG, "bmp width: " + bitmap.getWidth() + ", height = " + bitmap.getHeight());
        bitmap.getPixels(mIntValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < mDimImgWidth; ++i) {
            for (int j = 0; j < mDimImgHeight; ++j) {
                final int val = mIntValues[pixel++];
                mModelConfig.addImgValue(mImgData, val);
            }

        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    /** Prints top-K labels, to be shown in UI as the results. */
    private String printTopKLabels() {
        for (int i = 0; i < 10; ++i) {
            mSortedLabels.add(new AbstractMap.SimpleEntry<>(""+i, (mLabelProbArray[0][i]& 0xff) / 255.0f));
            if (mSortedLabels.size() > RESULTS_TO_SHOW) {
                mSortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = mSortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = mSortedLabels.poll();
            textToShow = String.format("\n%s   %4.8f",label.getKey(),label.getValue()) + textToShow;
        }
        return textToShow;
    }
}