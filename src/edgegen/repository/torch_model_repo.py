import torch
from pathlib import Path
from torch import nn
from pathlib import Path
from edgegen.repository.model_repo import ModelRepository

class PytorchModelRepository(ModelRepository):

    def __init__(self, model_folder: Path):
        super().__init__()
        self.model_folder = model_folder

    def save(self, model: nn.Module, model_name:str) -> None:
        """
        Save the model to the repository model folder.
        Using state dict approach.

        Parameters
        ----------
        model : nn.Module
            The model to save.
        """
        state_dict = model.state_dict()
        torch.save(state_dict, self.model_folder / (model_name + ".pt"))

    def load(self, architecture: nn.Module) -> nn.Module:
        """
        Load the model from the repository model folder.

        Parameters
        ----------
        architecture : nn.Module
            The architecture to be loaded

        Returns
        -------
        nn.Module
            The loaded model.
        """
        model_path = self.model_folder / f"{architecture.__class__.__name__}.pt"
        model = architecture.load_state_dict(torch.load(model_path, weights_only=True))
        return model
