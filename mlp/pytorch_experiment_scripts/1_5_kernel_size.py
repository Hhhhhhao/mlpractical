import numpy as np
from mlp import data_providers
from model_architectures import ConvolutionalNetwork
from experiment_builder  import ExperimentBuilder

batch_size = 128
image_num_channels = 1
image_height = 28
image_width = 28
num_filters = 128
num_layers = 4
experiment_names = ['3', '5', '7', '11']
kernel_sizes = [3, 5, 7, 11]
num_epochs = 50
weight_decay_coefficient =1e-5
seed = 7112018
use_gpu = True

rng = np.random.RandomState(seed=seed)
train_data = data_providers.EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
val_data = data_providers.EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
test_data = data_providers.EMNISTDataProvider('test', batch_size=batch_size, rng=rng)

for experiment_name, kernel_size in zip(experiment_names, kernel_sizes):
    print(experiment_name)

    custom_conv_net = ConvolutionalNetwork(
        input_shape=(batch_size, image_num_channels, image_height, image_width),
        dim_reduction_type=experiment_name,
        num_output_classes=47, num_filters=num_filters, num_layers=num_layers, kernel_size=kernel_size, use_bias=True)


    conv_experiment = ExperimentBuilder(
        network_model=custom_conv_net,
        experiment_name=experiment_name,
        num_epochs=num_epochs,
        weight_decay_coefficient=weight_decay_coefficient,
        use_gpu=use_gpu,
        train_data=train_data, val_data=val_data, test_data=test_data)

    total_losses = conv_experiment.run_experiment()