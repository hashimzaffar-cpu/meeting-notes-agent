from typing import Literal, Optional

from pydantic import BaseModel


class Action(BaseModel):
    task: str
    owner: Optional[str] = None
    due: Literal["today", "this_week", "this_month", "unspecified"]
    confidence: Literal["explicit", "inferred"]


class Output(BaseModel):
    actions: list[Action]
    unassigned: list[str]
