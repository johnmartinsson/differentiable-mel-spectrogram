import utils
import sys

def main():

    # ETC50
    config = {
        'dataset_name': 'esc50',
        'resample_rate': 8000,
    }

    base_data_dir = sys.argv[1]
    data_dir = base_data_dir + '/esc50'

    # get the dataset once to initialize dataset files
    trainset, validset, testset = utils.get_dataset_by_config(config, data_dir)
    print("ETC50 data. trainset: {}, validset: {}, testset: {}".format(len(trainset), len(validset), len(testset)))

    # Audio-MNIST
    config = {
        'dataset_name': 'audio_mnist',
    }

    data_dir = base_data_dir + '/audio_mnist'

    # get the dataset once to initialize dataset files
    trainset, validset, testset = utils.get_dataset_by_config(config, data_dir)
    print("Audio-MNIST data. trainset: {}, validset: {}, testset: {}".format(len(trainset), len(validset), len(testset)))


if __name__ == '__main__':
    main()