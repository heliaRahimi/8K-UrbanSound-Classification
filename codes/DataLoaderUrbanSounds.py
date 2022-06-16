import os

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tqdm import tqdm


class DataLoaderUrbanSounds():
    def __init__(self, input_dim):
        self.AUDIO_DIR = "UrbanSound8K/audio"
        self.METADATA = pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
        self.EXTRACTED_FEATURES = []
        self.INPUT_DIM = input_dim
        self.labelencoder = LabelEncoder()

    def __len__(self):
        return len(self.METADATA)

    def extract_features(self):
        for index_num, row in tqdm(self.METADATA.iterrows()):
            file_name = os.path.join(os.path.abspath(self.AUDIO_DIR), 'fold' + str(row["fold"]) + '/',
                                     str(row["slice_file_name"]))
            final_class_labels = row["class"]
            data = self.get_one_file_features_extractor(file_name)
            self.EXTRACTED_FEATURES.append([data, final_class_labels])

    def get_one_file_features_extractor(self, file_name):
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=self.INPUT_DIM)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features

    def get_train_test_data(self):
        ### converting extracted_features to Pandas dataframe
        extracted_features_df = pd.DataFrame(self.EXTRACTED_FEATURES, columns=['feature', 'class'])

        ### Split the dataset into independent and dependent dataset
        X = np.array(extracted_features_df['feature'].tolist())
        y = np.array(extracted_features_df['class'].tolist())

        return X, y

    def get_target_as_one_hot_encoder(self):
        ### converting extracted_features to Pandas dataframe
        extracted_features_df = pd.DataFrame(self.EXTRACTED_FEATURES, columns=['feature', 'class'])

        y = np.array(extracted_features_df['class'].tolist())
        y = to_categorical(self.labelencoder.fit_transform(y))
        return y

    def get_target_as_label_encoder(self):
        ### converting extracted_features to Pandas dataframe
        extracted_features_df = pd.DataFrame(self.EXTRACTED_FEATURES, columns=['feature', 'class'])

        y = np.array(extracted_features_df['class'].tolist())
        y = self.labelencoder.fit_transform(y)
        return y

    def split_to_train_test_data(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=50)
        return X_train, X_test, y_train, y_test

    def decode_label(self, predicted_label):
        return self.labelencoder.inverse_transform(predicted_label)

if __name__ == "__main__":
    print('Dataloader')
