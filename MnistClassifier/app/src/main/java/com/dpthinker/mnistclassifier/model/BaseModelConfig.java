package com.dpthinker.mnistclassifier.model;

import java.nio.ByteBuffer;

public abstract class BaseModelConfig {

    private int numBytesPerChannel = 4;

    private int dimBatchSize = 1;
    private int dimPixelSize = 3;

    private int dimImgWeight = 299;
    private int dimImgHeight = 299;

    private int imageMean = 0;
    private float imageSTD = 255.0f;

    private String modelName;

    public int getNumBytesPerChannel() {
        return numBytesPerChannel;
    }

    public int getDimBatchSize() {
        return dimBatchSize;
    }

    public int getDimPixelSize() {
        return dimPixelSize;
    }

    public int getDimImgWeight() {
        return dimImgWeight;
    }

    public int getDimImgHeight() {
        return dimImgHeight;
    }

    public String getModelName() {
        return modelName;
    }

    public int getImageMean() {
        return imageMean;
    }

    public float getImageSTD() {
        return imageSTD;
    }

    protected void setNumBytesPerChannel(int num) {
        this.numBytesPerChannel = num;
    }

    protected void setDimBatchSize(int size) {
        this.dimBatchSize = size;
    }

    protected void setDimPixelSize(int size) {
        this.dimPixelSize = size;
    }

    protected void setDimImgWeight(int weight) {
        this.dimImgWeight = weight;
    }

    protected void setDimImgHeight(int height) {
        this.dimImgHeight = height;
    }

    protected void setImageMean(int mean) {
        this.imageMean = mean;
    }

    protected void setImageSTD(float std) {
        this.imageSTD = std;
    }

    protected void setModelName(String name) {
        this.modelName = name;
    }

    protected abstract void setConfigs();

    public BaseModelConfig() {
        setConfigs();
    }

    public abstract void addImgValue(ByteBuffer buffer, int val);
}
