from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
from .models import ChatData, TestData
from .chat import introduce, ask, generate

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

@app.get("/")
def health_check():
    return {"status": "ok"}



@app.post("/")
def chat(data: ChatData):
    if len(data.messages) == 0:
        return StreamingResponse(introduce())
    return StreamingResponse(ask(data.messages))


@app.post("/test")
def test(data: TestData):
    return StreamingResponse(generate(data.prompt))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
