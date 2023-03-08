import logging
from pathlib import Path
from typing import NamedTuple

import pandas as pd


logger = logging.getLogger(__name__)


class UserData(NamedTuple):
    user_id: int
    texts: tuple[str, ...]


def _read_reviews_file(reviews_file_path: str) -> pd.DataFrame:
    """

    :param reviews_file_path: the location of the reviews file from the IMDb1m dataset
    :return:
    """
    reviews_file_parsed_path = Path(reviews_file_path)
    if reviews_file_parsed_path.name != "imdb1m-reviews.txt":
        logger.warning("The reviews file has a different name than the one in the original dataset.")

    # These are the column names of the reviews data from the IMDb1m dataset. The column names that the dataset authors
    # specified in the README file are incorrect.
    column_names = ["userId", "rating", "itemId", "title", "content"]

    df = pd.read_csv(reviews_file_path, names=column_names, sep="\t")

    return df


def _read_posts_file(posts_file_path: str) -> pd.DataFrame:
    """

    :param posts_file_path:
    :return:
    """
    posts_file_parsed_path = Path(posts_file_path)
    if posts_file_parsed_path.name != "imdb1m-posts.txt":
        logger.warning("The posts file has a different name than the one in the original dataset.")

    column_names = ["postId", "userId", "title", "content"]

    df = pd.read_csv(posts_file_path, names=column_names, sep="\t")
    return df


def _group_user_texts(user_texts: dict[int, list[str]]) -> tuple[UserData, ...]:
    """

    :param user_texts:
    :return:
    """
    user_ids = set(user_texts.keys())
    return tuple(UserData(user_id, tuple(user_texts[user_id])) for user_id in user_ids)


def preprocess(reviews_file_path: str, posts_file_path: str) -> tuple[UserData, ...]:
    """

    :param reviews_file_path:
    :param posts_file_path:
    :return:
    """
    user_texts: dict[int, list[str]] = {}

    def _parse_reviews_data() -> None:
        logger.info("Processing reviews...")
        reviews_data = _read_reviews_file(reviews_file_path=reviews_file_path)

        for review in reviews_data.itertuples(index=False, name="Review"):
            user_id = review.userId
            review_text = review.content

            if user_id not in user_texts:
                user_texts[user_id] = []

            user_texts[user_id].append(review_text)

        del reviews_data

    def _parse_posts_data() -> None:
        logger.info("Processing posts...")
        posts_data = _read_posts_file(posts_file_path=posts_file_path)

        for post in posts_data.itertuples(index=False, name="Post"):
            user_id = post.userId
            post_text = post.content

            if user_id not in user_texts:
                user_texts[user_id] = []

            user_texts[user_id].append(post_text)

        del posts_data

    _parse_reviews_data()
    _parse_posts_data()

    filtered_user_texts = _group_user_texts(user_texts=user_texts)
    del user_texts

    return filtered_user_texts
