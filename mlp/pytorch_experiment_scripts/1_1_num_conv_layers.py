import numpy as np

batch_size = 128
image_num_channels = 1
image_height = 28
image_width = 28
dim_reduction_type = 'max_pooling'
num_filters = 64
possible_num_layers = [2,3,4,5]
experiment_names = ['2_layers', '3_layers', '4_layers', '5_layers',]
num_epochs = 50
weight_decay_coefficient = 0
seed = 7112018
use_gpu = True

rng = np.random.RandomState(seed=seed)
train_data = data_providers.EMNISTDataProvider('train', batch_size=batch_size, rng=rng)
val_data = data_providers.EMNISTDataProvider('valid', batch_size=batch_size, rng=rng)
test_data = data_providers.EMNISTDataProvider('test', batch_size=batch_size, rng=rng)

for experiment_name, num_layers in zip(experiment_names, possible_num_layers):
    print(experiment_name)

	custom_conv_net = ConvolutionalNetwork(
        input_shape=(batch_size, image_num_channels, image_height, image_width),
    	dim_reduction_type=dim_reduction_type,
    	num_output_classes=47, num_filters=num_filters, num_layers=num_layers, use_bias=True)

	conv_experiment = ExperimentBuilder(
        network_model=custom_conv_net,
        experiment_name=experiment_name,
        num_epochs=num_epochs,
        weight_decay_coefficient=weight_decay_coefficient,
        use_gpu=use_gpu,
        train_data=train_data, val_data=val_data, test_data=test_data)
	total_losses = conv_experiment.run_experiment()