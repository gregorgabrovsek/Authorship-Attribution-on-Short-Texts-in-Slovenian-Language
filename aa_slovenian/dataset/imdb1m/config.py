import logging
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HuggingFaceConfig:
    namespace: str
    token: str
    dataset_repo_prefix: str
    dataset_repo_postfix: str


@dataclass(frozen=True)
class IMDbConfig:
    review_texts_file_path: str
    posts_texts_file_path: str
    output_folder: str
    top_n_users: list[int]
    train_test_split: float


def read_huggingface_configuration() -> HuggingFaceConfig:
    logger.info("Loading HuggingFace configuration")

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    huggingface_config = HuggingFaceConfig(**config['HuggingFace'])

    return huggingface_config


def read_imdb_configuration() -> IMDbConfig:
    logger.info("Loading IMDb configuration")

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    imdb_config = IMDbConfig(**config['IMDb'])

    return imdb_config
