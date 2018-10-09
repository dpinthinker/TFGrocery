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

    private static final String TAG = "MainActivity";
    private static final String MODEL_PATH = "mnist_cnn.tflite";

    private static final int DIM_BATCH_SIZE = 1;
    private static final int DIM_PIXEL_SIZE = 1;

    private static final int DIM_IMG_WIDTH = 28;
    private static final int DIM_IMG_HEIGHT = 28;

    private static final int RESULTS_TO_SHOW = 3;

    private Interpreter tflite = null;
    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private ByteBuffer imgData = null;

    /** Preallocated buffers for storing image data in. */
    private int[] intValues = new int[DIM_IMG_WIDTH * DIM_IMG_HEIGHT];

    private float[][] labelProbArray = new float[1][10];

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
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
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            tflite = new Interpreter(loadModelFile(this));
            imgData = ByteBuffer.allocateDirect(
                    4 * DIM_BATCH_SIZE * DIM_IMG_WIDTH * DIM_IMG_HEIGHT * DIM_PIXEL_SIZE);
            imgData.order(ByteOrder.nativeOrder());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        Bitmap bmp = getBitmapFromAssets("s3.png");
        convertBitmapToByteBuffer(bmp);
        Drawable d = new BitmapDrawable(getResources(), bmp);
        ImageView mnistImgView = findViewById(R.id.mnist_img);
        mnistImgView.setImageDrawable(d);

        tflite.run(imgData, labelProbArray);
        Log.e(TAG, "result size: " + labelProbArray[0].length);
        for (int i = 0; i < 10; i++) {
            Log.e(TAG, "index " + i + " prob is " + labelProbArray[0][i]);
        }

        TextView textView = findViewById(R.id.tv_prob);
        textView.setText(printTopKLabels());
    }

    public Bitmap getBitmapFromAssets(String fileName) {
        AssetManager assetManager = getAssets();

        InputStream istr = null;
        try {
            istr = assetManager.open(fileName);
        } catch (IOException e) {
            e.printStackTrace();
        }
        Bitmap bitmap = BitmapFactory.decodeStream(istr);

        return bitmap;
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        Log.d(TAG, "bmp width: " + bitmap.getWidth() + ", height = " + bitmap.getHeight());
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_WIDTH; ++i) {
            for (int j = 0; j < DIM_IMG_HEIGHT; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat(val);
            }

        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    /** Prints top-K labels, to be shown in UI as the results. */
    private String printTopKLabels() {
        for (int i = 0; i < 10; ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(""+i, labelProbArray[0][i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            textToShow = String.format("\nPrediction: %s   Probability: %4.2f",label.getKey(),label.getValue()) + textToShow;
        }
        return textToShow;
    }
}