package com.dpthinker.mnistclassifier.model;

import java.nio.ByteBuffer;

public class KerasQuantizedModelConfig extends KerasModelConfig {
    @Override
    protected void setConfigs() {
        setModelName("keras_mnist_quantized_uint8_0_1.tflite");

        //Quantized model has only one channel
        setNumBytesPerChannel(1);

        setDimBatchSize(1);
        setDimPixelSize(1);

        setDimImgWeight(28);
        setDimImgHeight(28);

        setImageMean(0);
        setImageSTD(255.0f);
    }

    @Override
    public void addImgValue(ByteBuffer imgData, int val) {
        imgData.put((byte) (val & 0xFF));
    }
}
