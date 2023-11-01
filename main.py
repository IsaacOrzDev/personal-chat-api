from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
# import asyncio
import replicate

app = FastAPI()


async def generate(prompt: str):
    print(prompt)
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


class Data(BaseModel):
    prompt: str


@app.post("/")
def read_root(data: Data):
    return StreamingResponse(generate(data.prompt))
