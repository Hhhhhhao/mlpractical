import numpy as np
from mlp import data_providers
from model_architectures import ConvolutionalNetwork
from experiment_builder  import ExperimentBuilder

batch_size = 128
image_num_channels = 1
image_height = 28
image_width = 28
num_epochs = 50
weight_decay_coefficient =1e-5
use_gpu = True

num_filters = [16, 32, 48, 64]
kernel_sizes = [3, 5, 7, 9]
num_layers = [2, 3, 4, 5]
seeds = [20180415，20181111, 20181115]
dim_reduction_types = ['max_pooling', 'avg_pooling', 'strided_convolution', 'dilated_convolution']


for seed in seeds:
    rng = np.random.RandomState(seed=seed)
    train_data = data_providers.EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
    val_data = data_providers.EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
    test_data = data_providers.EMNISTDataProvider('test', batch_size=batch_size, rng=rng)
    for dim_reduction_type in dim_reduction_types:
        for num_filter in num_filters:
            experiment_name = dim_reduction_type + '_filters' + str(num_filter) 
            custom_conv_net = ConvolutionalNetwork(
                    input_shape=(batch_size, image_num_channels, image_height, image_width),
                    dim_reduction_type=experiment_name,
                    num_output_classes=47, 
                    num_filters=num_filter, 
                    num_layers=4, 
                    kernel_size=3, 
                    use_bias=True)

            conv_experiment = ExperimentBuilder(
                    network_model=custom_conv_net,
                    experiment_name=experiment_name,
                    num_epochs=num_epochs,
                    weight_decay_coefficient=weight_decay_coefficient,
                    use_gpu=use_gpu,
                    train_data=train_data, val_data=val_data, test_data=test_data)

            total_losses = conv_experiment.run_experiment()

        for kernel_size in kernel_sizes:
            experiment_name = dim_reduction_type + '_kernel' + str(kernel_size)
            custom_conv_net = ConvolutionalNetwork(
                    input_shape=(batch_size, image_num_channels, image_height, image_width),
                    dim_reduction_type=experiment_name,
                    num_output_classes=47, 
                    num_filters=64, 
                    num_layers=4, 
                    kernel_size=kernel_size, 
                    use_bias=True)

            conv_experiment = ExperimentBuilder(
                    network_model=custom_conv_net,
                    experiment_name=experiment_name,
                    num_epochs=num_epochs,
                    weight_decay_coefficient=weight_decay_coefficient,
                    use_gpu=use_gpu,
                    train_data=train_data, val_data=val_data, test_data=test_data)

            total_losses = conv_experiment.run_experiment()

        for num_layer in num_layers:
            experiment_name = dim_reduction_type + '_kernel' + str(kernel_size)
            custom_conv_net = ConvolutionalNetwork(
                    input_shape=(batch_size, image_num_channels, image_height, image_width),
                    dim_reduction_type=experiment_name,
                    num_output_classes=47, 
                    num_filters=64, 
                    num_layers=num_layer, 
                    kernel_size=3, 
                    use_bias=True)

            conv_experiment = ExperimentBuilder(
                    network_model=custom_conv_net,
                    experiment_name=experiment_name,
                    num_epochs=num_epochs,
                    weight_decay_coefficient=weight_decay_coefficient,
                    use_gpu=use_gpu,
                    train_data=train_data, val_data=val_data, test_data=test_data)

            total_losses = conv_experiment.run_experiment()