import argparse
import os
import tqdm
import numpy as np
import utils
import torch

from ray import tune

def predict_test(df, data_dir, experiment_path):
    df['test_accuracy'] = 0
    #predictionss = []
    #labelss = []
    for row in tqdm.tqdm(df.iterrows()):
        config = utils.get_config_by_row(row)
        break

    _,_,testset = utils.get_dataset_by_config(config, data_dir)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

    print("making test predictions (takes a couple of minutes on GPU) ...")
    for row in tqdm.tqdm(df.iterrows()):
        idx = row[0]
        #labels, predictions = utils.get_predictions_by_row(row, data_dir, split='test', device='cuda:0')

        labels, predictions = utils.get_predictions_by_row_new(row, testloader, device='cuda:0')
        test_acc = np.mean(labels == predictions)
        df.at[idx, 'test_accuracy'] = test_acc
        
        #predictionss.append(predictions)
        #labelss.append(labels)

    dataset_name = os.path.basename(data_dir)

    df.to_csv(os.path.join(experiment_path, "{}.csv".format(dataset_name)))
    #predictionss = np.array(predictionss)
    #labelss = np.array(labelss)
    #np.save(os.path.join(experiment_path, "{}_predictionss.npy".format(dataset_name)), predictionss)
    #np.save(os.path.join(experiment_path, "{}_labelss.npy".format(dataset_name)), labelss)

    return df

def main():
    parser = argparse.ArgumentParser(description='Produce plots.')
    parser.add_argument('--experiment_path', help='The name of the experiment directory.', required=True, type=str)
    parser.add_argument('--data_dir', help='The path to the audio data directory.', required=True, type=str)
    args = parser.parse_args()

    tuner = tune.Tuner.restore(path=args.experiment_path)
    print(args.experiment_path)
    result = tuner.fit()
    df = result.get_dataframe()

    predict_test(df, args.data_dir, args.experiment_path)


if __name__ == '__main__':
    main()
