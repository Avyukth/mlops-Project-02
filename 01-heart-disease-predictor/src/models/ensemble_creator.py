from typing import Dict, List
import pandas as pd
import torch
import torch.nn as nn
from src.models.base_model import BaseModel
from src.ensemble.voting import VotingEnsemble
from src.ensemble.stacking import StackingEnsemble

class EnsembleCreator:
    def __init__(self, config):
        self.config = config

    def create_ensemble_models(self, models: Dict[str, BaseModel], x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, BaseModel]:
        voting_soft = VotingEnsemble(list(models.items()), voting='soft')
        voting_hard = VotingEnsemble(list(models.items()), voting='hard')
        stacking_logistic = StackingEnsemble(list(models.values()), meta_model=models['Logistic'])
        stacking_lgbm = StackingEnsemble(list(models.values()), meta_model=models['LGBM'])

        ensemble_models = {
            "Ensemble_Soft": voting_soft,
            "Ensemble_Hard": voting_hard,
            "Stacking_Logistic": stacking_logistic,
            "Stacking_LGBM": stacking_lgbm
        }

        # Fit the ensemble models
        for name, model in ensemble_models.items():
            print(f"Fitting {name}...")
            model.fit(x_train, y_train)

        return ensemble_models

    def export_ensemble_models_to_onnx(self, ensemble_models: Dict[str, BaseModel], output_dir: str):
        for name, model in ensemble_models.items():
            onnx_file_path = f"{output_dir}/{name.lower()}_model.onnx"
            self._export_ensemble_to_onnx(model, onnx_file_path, x_train.shape[1])
            print(f"Exported {name} ensemble model to ONNX format: {onnx_file_path}")

    def _export_ensemble_to_onnx(self, ensemble_model, file_path: str, input_size: int):
        class TorchEnsemble(nn.Module):
            def __init__(self, ensemble_model):
                super(TorchEnsemble, self).__init__()
                self.ensemble_model = ensemble_model

            def forward(self, x):
                # This is a placeholder implementation and needs to be adapted
                # based on the specific ensemble model type (Voting or Stacking)
                return torch.tensor(self.ensemble_model.predict(x.numpy()))

        torch_ensemble = TorchEnsemble(ensemble_model)
        dummy_input = torch.randn(1, input_size)
        torch.onnx.export(torch_ensemble, dummy_input, file_path, 
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})
