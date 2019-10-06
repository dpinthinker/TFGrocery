package com.dpthinker.mnistclassifier.model;

public class ModelConfigFactory {

    public final static String INCEPTION_MODEL = "inceptionv3_mnist.tflite";
    public final static String KERAS_MODEL = "keras_mnist.tflite";
    public final static String INCEPTION_QUANT_MODEL = "inception_quantized_model";
    public final static String KERAS_QUANT_MODEL = "keras_quant_model";
    public final static String SAVED_MODEL = "saved_model";

    public static BaseModelConfig getModelConfig(String model) {
        if (model == INCEPTION_MODEL) {
            return new InceptionModelConfig();
        } else if (model == KERAS_MODEL) {
            return new KerasModelConfig();
        } else if (model == INCEPTION_QUANT_MODEL) {
            return new InceptionQuantizedModelConfig();
        } else if (model == KERAS_QUANT_MODEL) {
            return new KerasQuantizedModelConfig();
        } else if (model == SAVED_MODEL) {
            return new SavedModelConfig();
        }
        return null;
    }
}
