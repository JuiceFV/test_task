import time
import argparse
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from qa import load_qa
from typing import Optional
import uvicorn

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", type=str, default="chroma_store")
    parser.add_argument("--top_k", type=int, default=5)
    args, _ = parser.parse_known_args()
    app.state.qa = load_qa(args.persist_dir, args.top_k)
    yield

app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

@app.get("/")
def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
def post_form(request: Request, question: str = Form(...), date: Optional[str] = Form(None)):
    start_time = time.time()
    filters: dict[str, list[str] | dict[str, str]] = {}
    if date:
        filters["date"] = {"$eq": date}

    qa = app.state.qa
    result = qa.invoke({"query": question, "filters": filters or None})
    answer = result["result"]
    docs = result["source_documents"]
    response_time = round((time.time() - start_time) * 1000, 2)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "question": question,
            "answer": answer,
            "sources": docs,
            "response_time": response_time,
        },
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
