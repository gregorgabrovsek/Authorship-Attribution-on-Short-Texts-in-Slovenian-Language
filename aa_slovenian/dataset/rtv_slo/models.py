from pydantic import BaseModel


class Rating(BaseModel):
    sum: int


class User(BaseModel):
    id: str
    uname: str
    link: str


class Replies(BaseModel):
    count: int


class ReplyTo(BaseModel):
    id: int
    content: str
    stamp: str
    link: str
    rating: Rating
    user: User
    replies: Replies
    self_id: int | None = None


class Comment(BaseModel):
    id: int
    self_id: int | None = None
    content: str
    stamp: str
    link: str
    rating: Rating
    user: User
    reply_to: ReplyTo | None = None
    replies: Replies | None = None


class Response(BaseModel):
    count: int
    news_id: int
    page: int
    per_page: int
    readonly: bool
    comments: list[Comment] | None


class CommentListBody(BaseModel):
    response: Response
