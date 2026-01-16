# Streamlit UI — Semantic Search + Data Contracts + PPO (Toy)

This is a teaching demo UI that lets you:
- Enter a query + role (state features)
- See policy probabilities over retrieval actions
- View results + contract warnings
- Provide feedback (manual) or auto-train
- See learning curves + console-style logs like:
  `Iter  10 | recent mean return (last 10 blocks) = ...`

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- Semantic search uses TF-IDF cosine similarity (stand-in for embeddings).
- PPO update is simplified to a single-state update (good for classroom intuition).
