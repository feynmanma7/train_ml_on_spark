FROM apache/spark:3.5.1

USER root

# 基础依赖（不常变动，缓存复用）
RUN pip install --no-cache-dir \
    pyspark \
    xgboost \
    pandas \
    pyarrow \
    numpy \
    scikit-learn

# Jupyter 相关（可能单独调整）
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab-lsp \
    python-lsp-server