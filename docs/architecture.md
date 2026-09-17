# Target Architecture

```text
NBA data provider
       |
       v
Ingestion job ---> PostgreSQL <--- Prediction results
       |                 ^
       v                 |
Feature builder ---> Training job ---> MLflow registry
                                          |
                                          v
Next.js frontend <--- FastAPI prediction service
                          |
                          v
                   Prometheus metrics
```

## What each component teaches

| Component | Purpose | Skill demonstrated |
| --- | --- | --- |
| Ingestion job | Fetch and validate completed games | reliable data pipelines |
| PostgreSQL | Persist raw games, features, predictions, and results | data modeling and migrations |
| Feature builder | Generate only pre-game signals | data leakage prevention |
| Training job | Train and compare models | reproducible experimentation |
| MLflow | Store parameters, metrics, and model versions | model lifecycle management |
| FastAPI | Serve predictions via HTTP | production inference APIs |
| Next.js | Display games and model quality | usable ML product delivery |
| Docker Compose | Run services together locally | containerized development |
| Kubernetes | Deploy scalable services and scheduled jobs | cloud-native operations |

## Important design decisions

- The model uses a chronological split, never random train/test splitting.
- Feature values for a game are calculated before that game is added to team history.
- The baseline model will remain in the project permanently as a benchmark.
- Betting lines and injury feeds are later enhancements; they should not block the first complete version.
