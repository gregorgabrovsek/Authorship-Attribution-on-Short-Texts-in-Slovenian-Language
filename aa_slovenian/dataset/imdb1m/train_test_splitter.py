import logging
import random

from aa_slovenian.dataset.imdb1m.preprocess import UserData
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class DatasetItem(BaseModel):
    text: str
    label: int


def train_test_split(
    grouped_texts_by_user: tuple[UserData, ...],
    split: float = 0.75
) -> tuple[tuple[DatasetItem, ...], tuple[DatasetItem, ...]]:
    assert 0 < split < 1
    logger.info(f"Creating train/test split for top {len(grouped_texts_by_user)} users")

    take_from_each_user = min(len(user_data.texts) for user_data in grouped_texts_by_user)
    grouped_texts_by_user: list[UserData] = [
        UserData(user_data.user_id, user_data.texts[:take_from_each_user])
        for user_data in grouped_texts_by_user
    ]

    training_set = []
    test_set = []

    for user_index, user_data in enumerate(grouped_texts_by_user):
        split_index = take_from_each_user * split
        for text_index, text in enumerate(user_data.texts):
            dataset_item = DatasetItem(text=text, label=user_index)
            if text_index < split_index:
                training_set.append(dataset_item)
            else:
                test_set.append(dataset_item)

    random.shuffle(training_set)
    random.shuffle(test_set)

    return tuple(training_set), tuple(test_set)
