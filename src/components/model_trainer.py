import torch
import torch.nn as nn
import os
from src.logger import logging
from src.exception import CustomException
import sys


class ModelTrainerConfig:
    def __init__(self,lr=1e-4,betas=(0.9,0.999),epochs=21,batch_size=10):
        self.lr,self.betas,self.epochs,self.batch_size = lr,betas,epochs,batch_size
        self.model_dir = os.path.join('artifacts', 'model.pth')
class ModelTrainer:
    def __init__(self, config=None):
        if config!=None:
            self.config = config
        else :
            self.config = ModelTrainerConfig()

    def train(self, model, dataloader):

        try :
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            epochs = self.config.epochs
            batch = self.config.batch_size

            mse = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(),lr=self.config.lr, betas=self.config.betas)

            logging.info("Model Training is started...")
            for epoch in range(epochs):
                running_loss=0.
                for i, data in enumerate(dataloader):
                    # Every data instance is an input + label pair
                    inputs, labels = data

                    # Zero your gradients for every batch!
                    optimizer.zero_grad()

                    # Make predictions for this batch
                    inputs = inputs.unsqueeze(1).to(device).to(torch.float)
                    # outputs = model(inputs)
                    out = model(inputs).to(torch.float)
                    target = labels.to(device).to(torch.float)


                    # Compute the loss and its gradients
                    # loss = loss_fn(outputs, labels)
                    loss = mse(out, target[:,:,:out.shape[-1]]).to(torch.float)
                    loss.backward()

                    # Adjust learning weights
                    optimizer.step()

                    # Gather data and report
                    running_loss += loss.item()
                    if i % 1000 == 999:
                        # print('batch {} loss: {}'.format(i + 1, ))
                        logging.info('batch {} loss: {}'.format(i + 1, running_loss))
                        running_loss = 0.
                        
            os.makedirs('artifacts', exist_ok=True)
            logging.info("Training has been finished. Saving model....")
            torch.save(model.state_dict(), self.config.model_dir)
            logging.info("Model has been saved successfully in {}".format(self.config.model_dir))
        except Exception as err:
            CustomException(err, sys)
        