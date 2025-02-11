from fastapi import FastAPI

app = FastAPI()


@app.get("/embeddings/{query}")
def get_embeddings(query: str):
    return {"response": f"Embeddings for {query}"}
