from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
# from mangum import Mangum
from dotenv import load_dotenv
import uvicorn
import asyncio
import replicate

app = FastAPI()

load_dotenv()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def split_text_by_length(text, length):
    return [text[i:i+length] for i in range(0, len(text), length)]

async def introduce():
    with open("./data/introduction.txt", "r") as file:
        message = file.read()
    lines = split_text_by_length(message, 80)
    for line in lines:
        yield line
        await asyncio.sleep(0.3)


async def generate(prompt: str):
    output = replicate.run(
        "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
        input={
            "prompt": prompt,
            "max_new_tokens": 500,
            "min_new_tokens": -1,
            "top_k": 50,
            "top_p": 1,
        },
        stream=True,
    )
    for item in output:
        yield item


class Message(BaseModel):
    role: str
    content: str


class ChatData(BaseModel):
    messages: List[Message]
    id: str
    previewToken: Optional[str] = None


class TestData(BaseModel):
    prompt: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/")
def chat(data: ChatData):
    if len(data.messages) == 0:
        return StreamingResponse(introduce())

    message = data.messages[-1].content
    return StreamingResponse(generate(message))


@app.post("/test")
def test(data: TestData):
    return StreamingResponse(generate(data.prompt))


# handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
