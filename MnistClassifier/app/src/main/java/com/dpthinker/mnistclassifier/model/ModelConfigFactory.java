package com.dpthinker.mnistclassifier.model;

public class ModelConfigFactory {

    public final static String INCEPTION_MODEL = "inceptionv3_mnist.tflite";
    public final static String KERAS_MODEL = "keras_mnist.tflite";

    public static BaseModelConfig getModelConfig(String model) {
        if (model == INCEPTION_MODEL) {
            return new InceptionModelConfig();
        } else if (model == KERAS_MODEL) {
            return new KerasModelConfig();
        }
        return null;
    }
}
