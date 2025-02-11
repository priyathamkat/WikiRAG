from contextlib import asynccontextmanager

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


class Embeddings(BaseModel):
    embeddings: list[list[float]]


@app.post("/embeddings/", response_model=Embeddings)
async def get_embeddings(request: EmbeddingsRequest) -> Embeddings:
    embeddings = embedding_model[0].encode(request.texts, task="text-matching")
    return Embeddings(embeddings=embeddings.tolist())
