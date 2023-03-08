import logging

from huggingface_hub import create_repo
from datasets import load_dataset

logger = logging.getLogger(__name__)


class HFHelper:

    def __init__(self, namespace: str, token: str) -> None:
        self.namespace = namespace
        self.token = token

    def upload_dataset(self, repo_name: str, files: dict[str, str]) -> None:
        """
        Upload the dataset to Huggingface Datasets.

        :param repo_name:
        :param files:
        :return:
        """
        logger.info(f"Creating the dataset for {repo_name}")
        dataset = load_dataset("json", data_files=files)
        logger.info(f"Uploading the dataset {repo_name}")
        dataset.push_to_hub(repo_name, token=self.token)

    def create_dataset_repo(self, repo_name: str, private: bool = True) -> str:
        """

        :param repo_name:
        :param private:
        :return:
        """
        full_repo_name = f"{self.namespace}/{repo_name}"
        logger.info(f"Creating the dataset repo {repo_name}")
        create_repo(full_repo_name, repo_type="dataset", private=private, token=self.token)

        return full_repo_name
