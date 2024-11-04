import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_data(start_val, end_val, num_samples):
    x_vals = np.linspace(start_val, end_val, num_samples)
    y_vals = np.linspace(start_val, end_val, num_samples)
    coordinates = np.column_stack((x_vals, y_vals))
    # coordinates = np.random.uniform(start_val, end_val, (num_samples, 2))
    f = (coordinates[:, 0] + coordinates[:, 1]) ** 2
    # f(x + y) = (x + y)^2
    return coordinates, f


def model_testing(model, neuron_num, layer_num=1, epochs_num=100, star_val=0, end_val=10, train_size=10000, test_size=150, batch_size=50):
    coordinates_train, f_train = generate_data(star_val, end_val, train_size)
    model.compile(optimizer='adam', loss='mse')
    model.fit(coordinates_train, f_train, epochs=epochs_num, batch_size=batch_size)
    coordinates_test, f_test = generate_data(star_val, end_val, test_size)
    predictions = model.predict(coordinates_test)
    margin_list = [abs(f_test[i] - predictions[i][0]) for i in range(test_size)]
    plt.subplot(211)
    plt.plot(margin_list)
    plt.xlabel("Loss")
    plt.title(f"Error plot for model with {layer_num} inner layer and {neuron_num} neurons")
    plt.subplot(212)
    plt.plot(f_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    print(f"Average arithmetic error: {sum(margin_list) / test_size}")
    plt.show()


def feed_forward(neuron_num):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(neuron_num, input_dim=2, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

def cascade_forward(neuron_num, layers=1):
    input_layer = tf.keras.Input(shape=(2,), name='input')
    current = tf.keras.layers.Dense(neuron_num, activation='relu', input_shape=(1,))(input_layer)
    for i in range(layers-1):
        concatenated_layer = tf.keras.layers.concatenate([input_layer, current])
        current = tf.keras.layers.Dense(neuron_num, activation='relu', input_shape=(1,))(concatenated_layer)
    output_layer = tf.keras.layers.Dense(1, name='output')(current)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

def elman(neuron_num, layers=1):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((1, 2), input_shape=(2,), name='input_reshape'))
    model.add(tf.keras.layers.SimpleRNN(neuron_num, return_sequences=True, activation='relu', input_shape=(1, 2)))
    for i in range(layers-1):
        model.add(tf.keras.layers.SimpleRNN(neuron_num, return_sequences=True, activation='relu'))
    model.add(tf.keras.layers.Dense(1, name='output'))
    model.add(tf.keras.layers.Reshape((1,), input_shape=(1, 1), name='output_reshape'))
    return model


model_testing(feed_forward(neuron_num=10), neuron_num=10)
model_testing(feed_forward(neuron_num=20), neuron_num=20)
model_testing(cascade_forward(neuron_num=20), neuron_num=20)
model_testing(cascade_forward(neuron_num=20, layers=2), layer_num=2, neuron_num=10)
model_testing(elman(neuron_num=15), neuron_num=15)
model_testing(elman(neuron_num=5, layers=3), layer_num=3, neuron_num=5)