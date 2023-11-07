from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str


class ChatData(BaseModel):
    messages: List[Message]
    id: str
    previewToken: Optional[str] = None


class TestData(BaseModel):
    prompt: str
