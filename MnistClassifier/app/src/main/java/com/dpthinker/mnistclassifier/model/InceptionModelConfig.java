package com.dpthinker.mnistclassifier.model;

import java.nio.ByteBuffer;

public class InceptionModelConfig extends BaseModelConfig {

    @Override
    protected void setConfigs() {
        setModelName("inceptionv3_mnist.tflite");

        setDimBatchSize(1);
        setDimPixelSize(3);

        setDimImgWeight(299);
        setDimImgHeight(299);

        setImageMean(0);
        setImageSTD(255.0f);
    }

    public void addImgValue(ByteBuffer imgData, int val) {
        imgData.putFloat((((val >> 16) & 0xFF) - getImageMean()) / getImageSTD());
        imgData.putFloat((((val >> 8) & 0xFF) - getImageMean()) / getImageSTD());
        imgData.putFloat(((val & 0xFF) - getImageMean()) / getImageSTD());
    }
}
