from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel

embedding_model = []


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load embedding model
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
    embedding_model.append(model)
    yield
    # Unload embedding model
    embedding_model.clear()


app = FastAPI(lifespan=lifespan)


class EmbeddingsRequest(BaseModel):
    texts: list[str]
    task_type: Literal["query", "passage"]


class Embeddings(BaseModel):
    embeddings: list[list[float]]


@app.post("/embeddings/", response_model=Embeddings)
async def get_embeddings(request: EmbeddingsRequest) -> Embeddings:
    task = None
    match request.task_type:
        case "query":
            task = "retrieval.query"
        case "passage":
            task = "retrieval.passage"
    embeddings = embedding_model[0].encode(request.texts, task=task)
    return Embeddings(embeddings=embeddings.tolist())
