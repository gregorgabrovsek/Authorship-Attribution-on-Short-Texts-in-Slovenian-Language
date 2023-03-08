import logging
import os

import json
from pydantic.json import pydantic_encoder

from aa_slovenian.dataset.imdb1m.train_test_splitter import DatasetItem

logger = logging.getLogger(__name__)


def output_to_file(user_data: tuple[DatasetItem, ...], prefix: str, n: int, output_folder: str) -> str:
    json_data = json.dumps(user_data, default=pydantic_encoder)

    filename = os.path.join(output_folder, f"{prefix}_top{n}.json")
    logger.info(f"Writing {prefix} set")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(json_data)

    return filename
