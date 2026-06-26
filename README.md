# RadRAG

Multimodal RAG system for radiology. Upload a medical image, get a description grounded in retrieved papers, with citations.

**Status:** Work in progress.

## Stack
- FastAPI · Anthropic Claude · Qdrant · sentence-transformers · LangGraph
- Python 3.11+, Docker

## Setup
```bash
cp .env.example .env  # fill in your ANTHROPIC_API_KEY
docker compose up -d  # start Qdrant
uv sync               # install Python deps
```

## Roadmap
- [x] Project skeleton
- [ ] Paper ingestion + embedding
- [ ] Retrieval + generation API
- [ ] Multimodal (image → description → retrieval)
- [ ] Agent layer (LangGraph)
- [ ] Eval harness (RAGAS)
- [ ] Streamlit UI
- [ ] Deployed demo
