
from src.components.data_preparation import DataPreparation
from src.model import WaveUNet
import torch
from src.components.model_trainer import ModelTrainer


if __name__ == '__main__':
    preparation = DataPreparation()
    dataloader = preparation.transform()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WaveUNet(num_channel=1, num_sources=5).to(device)
    trainer = ModelTrainer()
    trainer.train(model=model, dataloader=dataloader)