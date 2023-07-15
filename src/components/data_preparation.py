import musdb
import librosa
import numpy as np
import torch

from src.logger import logging
from src.exception import CustomException
import sys, os

class DataPreparationConfig:
    def __init__(self, batch_size=64, shuffle=True):
        self.batch_size, self.shuffle = batch_size, shuffle


class DataPreparation():
    def __init__(self, batch_size=64, shuffle=True, config=None):
        if config == None:
            self.config = DataPreparationConfig(batch_size, shuffle)
        else :
            self.config = config
        
        self.ingestion()
    
    def ingestion(self):
        try :
            os.makedirs("artifacts", exist_ok=True)
            logging.info("Data Ingestion has started. Downloading the dataset from MUSDB repository")
            self.train = musdb.DB(download=True, subsets='train', root="artifacts")
            self.test = musdb.DB(download=True, subsets='test', root="artifacts")
            logging.info("MUSDB dataset has been successfully downloaded!")
            logging.info("Data Ingestion process is finished")
        except Exception as err:
            CustomException(err, sys)
    
    def transform(self):
        try :
            logging.info("Data Transformation has started...")

            target_names = ['vocals','accompaniment','drums','bass','other']
            feature_name = 'linear_mixture'

            X_l = []
            Y_l = []

            for track in self.train:
                X_j = []
                for target in target_names:
                    X_j.append(librosa.to_mono(track.targets[target].audio.T))
                X_l.append(librosa.to_mono(track.targets[feature_name].audio.T))
                Y_l.append(X_j)
            X = np.array(X_l)
            Y = np.array(Y_l)

            logging.info("Data Transformation process is finished")
            
            return self.build_dataloader(X, Y)
        except Exception as err:
            CustomException(err, sys)

                    
    def build_dataloader(self,X, Y):
        X_train = torch.from_numpy(X)
        Y_train = torch.from_numpy(Y)
        dataset = torch.utils.data.TensorDataset(Y_train, X_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        return dataloader