1. Run create_csv.py to create the CSV file containing the labels of each image in the dataset
2. Run sampling.py to create the CSV files with the labels, but after splitting the images for training, testing and validation.
3. Change values in config.py as required.
4. Change transformations in custom_dataloader.py to suit the dataset.
5. Run hyperparameters.py to find the best hyperparameters for training.
6. Run train.py to train the model for up to 50 epochs. The model's optimal state (number of epochs and least epoch loss) is saved to a folder named 'checkpoints'.
7. Run inference.py to predict the age of a face.

Please refer to Mousavi's requirements.txt to find a clean list of required Python packages. Not all packages in packagelist.txt are used.

Remember to download the supporting files utils.py, functions.py.

Remember to change any Paths/directories in the files for your own use.
