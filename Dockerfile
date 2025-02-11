FROM python:3.13-slim

# Install `curl`
RUN apt update && apt install -y curl

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY . .

# Install package
RUN poetry install

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "wiki_rag.servers.embeddings:app", "--host", "0.0.0.0", "--port", "8000"]
