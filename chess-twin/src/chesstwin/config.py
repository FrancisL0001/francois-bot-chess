"""

Loads the configuration from ./configs/default.yaml and parses it into a pydantic model that mimics
the configuration structure. 

"""

from pydantic import BaseModel, ConfigDict
from typing import Optional

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True)

class ProjectConfig(StrictModel):
    name : str
    seed : int

class PlayerConfig(StrictModel):
    username : str
    time_control : str
    recency_cutoff_months: int

class DataConfig(StrictModel):
    raw_dir : str
    processed_dir : str
    split : str
    test_fraction: float
    val_fraction : float

class EncodingConfig(StrictModel):
    move_vocab : str
    num_board_planes : int

class ModelConfig(StrictModel):
    trunk : str
    num_blocks : int
    num_filters : int

class TrainingConfig(StrictModel):
    device : str
    batch_size : int
    optimizer : str
    lr : float
    epochs : int
    early_stop_metric : str
    patience : int

class TransferConfig(StrictModel):
    pretrain  : bool
    pretrain_rating_band : list[Optional[int]]
    finetune_lr_factor : float

class EvalConfig(StrictModel):
    stockfish_path : str
    analysis_depth : int
    blunder_threshold_cp : int

class ServingConfig(StrictModel):
    artifact_format : str
    latency_budget_ms : int
    model_version : str


class Config(BaseModel):
    """
    The configuration model for the chess twin application. This model is used to parse the configuration
    from the YAML file and provide type safety when accessing configuration values.
    """
    project : ProjectConfig
    player : PlayerConfig
    data : DataConfig
    encoding : EncodingConfig
    model: ModelConfig
    training : TrainingConfig
    transfer : TransferConfig
    eval_ : EvalConfig
    serving : ServingConfig



# Loading the configuratio from the YAML file and parsing it

import yaml 

def load_config(config_path : str) -> Config:
    """
    Loads the configuration from the specified YAML file and parses it into a Config object.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        Config: The parsed configuration object.
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)
