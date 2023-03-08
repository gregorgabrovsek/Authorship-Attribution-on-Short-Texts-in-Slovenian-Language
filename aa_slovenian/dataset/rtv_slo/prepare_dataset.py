import math
import os
import re
import json
import random

DATA_BASE_FOLDER = "rtv_scraped_data"
OUTPUT_BASE_FOLDER = "rtv_datasets"

NUMBER_OF_USERS = 200
TRAIN_TEST_SPLIT = 0.75


def process_comment_string(comment: str) -> str:
    comm: str = comment["c"]
    comm = comm.replace("\r\n", "\n")
    comm = re.sub(r"(@)?.{1,30}\s(# )?\d{2}\.\d{2}.\d{4} ob \d{2}:\d{2}", "", comm)
    comm = re.sub(r'\[url=\"http.*?\"\].*?\[\/url\]', "URL", comm)
    comm = re.sub(r'((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', "URL", comm)
    comm = re.sub(r'<p>', " ", comm)
    comm = re.sub(r'</p>', " ", comm)
    comm = re.sub(r'<br>', " ", comm)
    comm = re.sub(r'\[i\]', "", comm)
    comm = re.sub(r'\[b\]', "", comm)
    comm = re.sub(r'\[\\i\]', "", comm)
    comm = re.sub(r'\[\\b\]', "", comm)
    comm = re.sub(r'\[\/i\]', "", comm)
    comm = re.sub(r'\[\/b\]', "", comm)
    comm = comm.replace("\n\n", "\n")
    comm = re.sub(r'\n', " ", comm)
    for _ in range(10):
        comm = re.sub(r'  ', " ", comm)
    comm = comm.strip()
    return comm


def prepare_prepare_dataset(number_of_included_users: int, add_out_of_class: bool):
    print(
        f"Preparing dataset for {number_of_included_users} users with{'out' if add_out_of_class else ''} out-of-class comments.")

    # Get the minimum number of comments for a user
    min_comments_for_user = 10e10
    for i in range(number_of_included_users):
        with open(
            os.path.join(DATA_BASE_FOLDER, "user_comments", f"{i}.json"),
            "r",
            encoding="utf-8",
        ) as comment_file:
            data = json.loads(comment_file.read())
        min_comments_for_user = min(min_comments_for_user, len(data))
    min_comments_for_user = int(min_comments_for_user)

    split_index = min_comments_for_user * TRAIN_TEST_SPLIT

    # Get the comments for each user

    final_train_data = []
    final_test_data = []

    for i in range(number_of_included_users):
        with open(
            os.path.join(DATA_BASE_FOLDER, "user_comments", f"{i}.json"),
            "r",
            encoding="utf-8",
        ) as comment_file:
            data = json.loads(comment_file.read())
        random.shuffle(data)
        data = data[:min_comments_for_user]

        preprocessed_comments = []
        for comm in data:
            preprocessed_comments.append(process_comment_string(comm))

        for index, comment in enumerate(preprocessed_comments):
            data_item = {"text": comment, "label": i}

            if index <= split_index:
                final_train_data.append(data_item)
            else:
                final_test_data.append(data_item)

    if add_out_of_class:
        comments_per_user = math.ceil(min_comments_for_user / (NUMBER_OF_USERS - number_of_included_users))
        out_of_class_comments = []
        for i in range(number_of_included_users, NUMBER_OF_USERS):
            with open(
                os.path.join(DATA_BASE_FOLDER, "user_comments", f"{i}.json"),
                "r",
                encoding="utf-8",
            ) as comment_file:
                data = json.loads(comment_file.read())
            random.shuffle(data)
            data = data[:comments_per_user]

            preprocessed_comments = list(map(process_comment_string, data))

            for index, comment in enumerate(preprocessed_comments):
                data_item = {"text": comment, "label": number_of_included_users}
                out_of_class_comments.append(data_item)
        random.shuffle(out_of_class_comments)
        out_of_class_comments = out_of_class_comments[:min_comments_for_user]
        final_train_data += out_of_class_comments[:int(split_index + 1)]
        final_test_data += out_of_class_comments[int(split_index + 1):]

    random.shuffle(final_train_data)
    random.shuffle(final_test_data)

    # Check if the OUTPUT_BASE_FOLDER exists
    if not os.path.exists(OUTPUT_BASE_FOLDER):
        os.makedirs(OUTPUT_BASE_FOLDER)

    folder_name = os.path.join(
        OUTPUT_BASE_FOLDER,
        f"RTVCommentsTop{number_of_included_users}Users{'WithOOC' if add_out_of_class else 'WithoutOOC'}"
    )
    os.makedirs(folder_name, exist_ok=True)

    with open(
        os.path.join(
            folder_name,
            f"train_TOP{number_of_included_users}_{'WITHOOC' if add_out_of_class else 'NOOOC'}.json"
        ),
        "w"
    ) as jsonfile:
        json.dump(final_train_data, jsonfile)

    with open(
        os.path.join(
            folder_name,
            f"test_TOP{number_of_included_users}_{'WITHOOC' if add_out_of_class else 'NOOOC'}.json"
        ),
        "w"
    ) as jsonfile:
        json.dump(final_test_data, jsonfile)


if __name__ == '__main__':
    for include_ooc in [True, False]:
        for i in [5, 10, 20, 50, 100]:
            prepare_prepare_dataset(i, include_ooc)
