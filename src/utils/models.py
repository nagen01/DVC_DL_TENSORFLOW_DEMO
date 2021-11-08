import tensorflow as tf
import logging

def get_VGG_16_model(input_shape, model_path):
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )

    model.save(model_path)
    logging.info(f"VGG16 base model saved at {model_path}")
    return model