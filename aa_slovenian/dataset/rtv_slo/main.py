import time
from typing import Any

import requests
import json
from alive_progress import alive_bar
from pathlib import Path
import os

from aa_slovenian.dataset.rtv_slo.models import CommentListBody, Response, Comment

BASE_RTV_URL = "https://www.rtvslo.si"
BASE_COMMENT_URL = "https://api.rtvslo.si/comments"

headers = {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0"}
BASE_FOLDER = "rtv_scraped_data"


class TooEarlyException(Exception):
    pass


FIRST_ARTICLE_ID = 400000
LAST_ARTICLE_ID = 650719
NUMBER_OF_USERS = 200


def scrape_from_rtv_slo(do_not_attempt: set[str]):
    article_ids = range(FIRST_ARTICLE_ID, LAST_ARTICLE_ID + 1)

    # Check if BASE_FOLDER exists, if not, create it
    if not Path(BASE_FOLDER).is_dir():
        os.mkdir(BASE_FOLDER)
        os.mkdir(os.path.join(BASE_FOLDER, "comments"))

    with requests.session() as session:
        session.get(BASE_RTV_URL, headers=headers)

        with alive_bar(len(article_ids), force_tty=True) as bar:
            for index, id in enumerate(article_ids):
                if id in do_not_attempt:
                    time.sleep(5)
                    continue

                try:
                    if Path(os.path.join(BASE_FOLDER, "comments", f"{id}.json")).is_file():
                        bar()
                        continue

                    comment_page = 0
                    comments: list[Comment] = []
                    while True:
                        comment_list_url = f"{BASE_COMMENT_URL}/{id}/list?sort=rating&order=desc&pageNumber={comment_page}&pageSize=100"
                        data: Response = CommentListBody(**session.get(
                            url=comment_list_url,
                            headers=headers
                        ).json()).response

                        if data.comments is None:
                            break

                        for comment in data.comments:
                            comments.append(comment)

                        comment_page += 1

                    with open(os.path.join(BASE_FOLDER, "comments", f"{id}.json"), "w",
                              encoding="utf-8") as comment_file:
                        comment_file.write(json.dumps(comments, default=lambda x: x.dict()))

                    bar()
                except Exception:
                    raise TooEarlyException(f"rtv id {id}")
        return True


def get_user_stats():
    article_ids = range(FIRST_ARTICLE_ID, LAST_ARTICLE_ID + 1)
    user_stats: dict[str, int] = {}
    with alive_bar(len(article_ids), force_tty=True) as bar:
        for id in article_ids:
            with open(os.path.join(BASE_FOLDER, "comments", f"{id}.json"), "r", encoding="utf-8") as comment_file:
                data = json.loads(comment_file.read())
            for comment in data:
                user_id = comment["user"]["id"]
                user_stats[user_id] = user_stats.get(user_id, 0) + 1
            bar()
    converted_stats = [{"userID": uid, "numberOfComments": occ} for uid, occ in user_stats.items()]
    converted_stats.sort(key=lambda x: -x["numberOfComments"])
    with open(os.path.join(BASE_FOLDER, "user_stats.json"), "w", encoding="utf-8") as output_file:
        output_file.write(json.dumps(converted_stats))


def retrieve_user_specific_data():
    with open(os.path.join(BASE_FOLDER, "user_stats.json"), "r", encoding="utf-8") as user_data:
        data = json.loads(user_data.read())

    requested_user_ids = {data[i]["userID"] for i in range(NUMBER_OF_USERS)}
    user_id_to_leaderboard_place = {data[i]["userID"]: i for i in range(NUMBER_OF_USERS)}
    user_comments: dict[str, list[dict[str, Any]]] = {data[i]["userID"]: [] for i in range(NUMBER_OF_USERS)}

    article_ids = range(FIRST_ARTICLE_ID, LAST_ARTICLE_ID + 1)
    print("Getting data from comment data...")
    with alive_bar(len(article_ids), force_tty=True) as bar:
        for id in article_ids:
            with open(os.path.join(BASE_FOLDER, "comments", f"{id}.json"), "r", encoding="utf-8") as comment_file:
                data = json.loads(comment_file.read())
            for comment in data:
                user_id = comment["user"]["id"]
                if user_id in requested_user_ids:
                    user_comments[user_id].append({
                        "id": comment["id"],
                        "c": comment["content"],
                        "t": comment["stamp"],
                        "r": comment["rating"]["sum"],
                    })
            bar()

    print("Saving user comments...")
    # Check if the folder exists
    if not os.path.exists(os.path.join(BASE_FOLDER, "user_comments")):
        os.mkdir(os.path.join(BASE_FOLDER, "user_comments"))
    with alive_bar(len(requested_user_ids), force_tty=True) as foo:
        for user_id in requested_user_ids:
            with open(
                os.path.join(BASE_FOLDER, "user_comments", f"{user_id_to_leaderboard_place[user_id]}.json"),
                "w",
                encoding="utf-8",
            ) as comment_file:
                comment_file.write(json.dumps(user_comments[user_id]))
            foo()


if __name__ == '__main__':
    final_resp = False
    failed: dict[str, int] = {}
    while not final_resp:
        try:
            final_resp = scrape_from_rtv_slo(set([x for x, times in failed.items() if times > 10]))
        except Exception as e:
            print(e)
            id = str(e).split(" ")[-1]
            failed[id] = failed.get(id, 0) + 1
    get_user_stats()
    retrieve_user_specific_data()


