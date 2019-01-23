package com.dpthinker.mnistclassifier.model;

import java.nio.ByteBuffer;

public class InceptionQuantizedModelConfig extends InceptionModelConfig {
    @Override
    protected void setConfigs() {
        setModelName("inceptionv3_mnist_quantized_uint8.tflite");

        setNumBytesPerChannel(1);

        setDimBatchSize(1);
        setDimPixelSize(3);

        setDimImgWeight(299);
        setDimImgHeight(299);

        setImageMean(0);
        setImageSTD(255.0f);
    }

    @Override
    public void addImgValue(ByteBuffer imgData, int val) {
        imgData.put((byte) ((((val >> 16) & 0xFF) - getImageMean()) / getImageSTD()));
        imgData.put((byte) ((((val >> 8) & 0xFF) - getImageMean()) / getImageSTD()));
        imgData.put((byte) (((val & 0xFF) - getImageMean()) / getImageSTD()));
    }
}