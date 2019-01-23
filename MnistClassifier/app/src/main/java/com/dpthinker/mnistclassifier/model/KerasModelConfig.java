package com.dpthinker.mnistclassifier.model;

import java.nio.ByteBuffer;

public class KerasModelConfig extends BaseModelConfig {

    @Override
    protected void setConfigs() {
        setModelName("keras_mnist.tflite");

        setDimBatchSize(1);
        setDimPixelSize(1);

        setDimImgWeight(28);
        setDimImgHeight(28);

        setImageMean(0);
        setImageSTD(255.0f);
    }

    public void addImgValue(ByteBuffer imgData, int val) {
        imgData.putFloat(((val & 0xFF) - getImageMean()) / getImageSTD());
    }
}
