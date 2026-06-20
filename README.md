# 1. Build Docker Image
Including xgboost and spark and jupyter lab.

```bash
docker build -t spark-xgb:latest .
```

# 2. Start Docker Compose
```bash
docker-compose up -d
```

# 3. Train XGBoost Model on spark distributed cluster
