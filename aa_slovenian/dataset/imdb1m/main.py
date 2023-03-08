import logging

from aa_slovenian.dataset.huggingface.helper import HFHelper
from aa_slovenian.dataset.imdb1m.config import (
    read_huggingface_configuration,
    read_imdb_configuration
)
from aa_slovenian.dataset.imdb1m.helpers import output_to_file
from aa_slovenian.dataset.imdb1m.preprocess import preprocess, UserData
from aa_slovenian.dataset.imdb1m.train_test_splitter import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

# create logger
logger = logging.getLogger(__name__)


def main() -> None:
    huggingface_config = read_huggingface_configuration()
    imdb_config = read_imdb_configuration()

    imdb_texts_by_user = preprocess(
        posts_file_path=imdb_config.posts_texts_file_path,
        reviews_file_path=imdb_config.review_texts_file_path,
    )

    def get_len(x: UserData) -> int:
        return len(x.texts)

    sorted_texts_by_user = sorted(imdb_texts_by_user, key=lambda x: -get_len(x))
    hf_helper = HFHelper(namespace=huggingface_config.namespace, token=huggingface_config.token)

    for top_n in sorted(imdb_config.top_n_users, key=lambda x: -x):
        sorted_texts_by_user = sorted_texts_by_user[:top_n]
        training_set, test_set = train_test_split(
            grouped_texts_by_user=sorted_texts_by_user,
            split=imdb_config.train_test_split
        )

        files = {
            "train": output_to_file(
                user_data=training_set, prefix="train", n=top_n, output_folder=imdb_config.output_folder
            ),
            "test": output_to_file(
                user_data=test_set, prefix="test", n=top_n, output_folder=imdb_config.output_folder
            ),
        }

        full_repo_name = hf_helper.create_dataset_repo(
            repo_name=f"{huggingface_config.dataset_repo_prefix}-{top_n}-{huggingface_config.dataset_repo_postfix}"
        )
        hf_helper.upload_dataset(repo_name=full_repo_name, files=files)


if __name__ == '__main__':
    main()
