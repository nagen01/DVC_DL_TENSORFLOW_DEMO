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

def prepare_model(model, CLASSES, freez_all, freez_till, learning_rate):
    if freez_all:
        for layer in model.layers:
            layer.trainable = False
    elif(freez_till is not None) and (freez_till > 0):
        for layer in model.layers[:-freez_till]:
            layer.trainable = False
        
    ## Add our fully connected layers
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(units=CLASSES, activation="softmax")(flatten_in)
    full_model = tf.keras.models.Model(inputs = model.input, outputs = prediction)
    full_model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]
    )

    logging.info("custom model is compiled and ready to be trained")
    full_model.summary()
    return full_model

