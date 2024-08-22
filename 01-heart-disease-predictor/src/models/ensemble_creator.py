from typing import Dict, List
import pandas as pd
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
