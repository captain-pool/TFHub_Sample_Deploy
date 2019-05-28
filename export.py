import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class MNIST(tf.keras.models.Model):
  def __init__(self, output_activation="softmax"):
    super(MNIST, self).__init__()
    self.layer_1 = tf.keras.layers.Dense(32)
    self.layer_2 = tf.keras.layers.Dense(10, activation=output_activation)
  
  @tf.function(input_signature=[tf.TensorSpec(shape=[None, 28, 28, 1], dtype=tf.uint8)])
  def call(self, inputs):
    casted = tf.cast(inputs, tf.float32)
    flatten = tf.keras.layers.Flatten()(casted)
    normalize = tf.keras.layers.Lambda(lambda x:x / tf.reduce_max(tf.gather(x, 0)))(flatten)
    x = self.layer_1(normalize)
    x = self.layer_2(x)
    output = tf.keras.layers.Lambda(lambda x: tf.argmax(x, axis=1))(x)
    return output

model = MNIST()
train, test = tfds.load("mnist", split=["train", "test"])
optimizer_fn = tf.optimizers.Adam(learning_rate = 1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()
metric = tf.keras.metrics.Mean()
model.compile(optimizer_fn, loss=loss_fn)
train = train.batch(32)


@tf.function
def train_step(image, label):
  with tf.GradientTape() as tape:
    preds = model(image)
    loss_ = loss_fn(label, preds)
  grads = tape.gradient(loss_, model.trainable_variables)
  optimizer_fn.apply_gradients(zip(grads, model.trainable_variables))
  metric(loss_)


for epoch in range(3):
  for step, data in enumerate(train):
    train_step(data['image'], data['label'])
    if step % 100 == 0:
      print("Epoch #{}::\tStep #{}:\tLoss: {}".format(epoch, step, metric(loss).numpy()))

tf.saved_model.save(model, "mnist/digits/1")
