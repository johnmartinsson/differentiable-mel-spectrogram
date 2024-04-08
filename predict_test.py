import argparse
import os
import tqdm
import numpy as np
import utils
import torch

from ray import tune

def predict_test(df, data_base_dir, ray_results_dir, dataset_name):
    df['test_accuracy'] = 0
    #predictionss = []
    #labelss = []
    for row in tqdm.tqdm(df.iterrows()):
        config = utils.get_config_by_row(row)
        break

    
    _,_,testset = utils.get_dataset_by_config(config, os.path.join(data_base_dir, dataset_name))
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

    experiment_path = os.path.join(ray_results_dir, dataset_name)
    print("saving predictions to file {} ...".format(os.path.join(experiment_path, "{}.csv")))
    df.to_csv(os.path.join(experiment_path, "{}.csv".format(dataset_name)))

    return df

def main():
    parser = argparse.ArgumentParser(description='Produce plots.')
    parser.add_argument('--ray_results_dir', help='The name of the ray results directory.', required=True, type=str)
    parser.add_argument('--data_base_dir', help='The path to the audio data directory.', required=True, type=str)
    parser.add_argument('--dataset_name', help='The dataset name.', required=True, type=str)
    args = parser.parse_args()

    experiment_path = os.path.join(args.ray_results_dir, args.dataset_name)
    tuner = tune.Tuner.restore(path=experiment_path)
    result = tuner.fit()
    df = result.get_dataframe()

    df = predict_test(df, args.data_base_dir, args.ray_results_dir, args.dataset_name)

    print(df.head())


if __name__ == '__main__':
    main()
