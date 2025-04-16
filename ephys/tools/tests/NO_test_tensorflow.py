# test for tensorflow install.

try:
    import tensorflow as tf
except:
    raise ImportError("tensorflow is not installed.")
print(f"Testing tensorflow version: {tf.__version__}")  
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
print("   Tensorflow - starting fit")
model.fit(x_train, y_train, epochs=5, batch_size=64)
print("   Tensorflow test complete.")