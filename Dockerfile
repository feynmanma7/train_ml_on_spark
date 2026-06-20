FROM apache/spark:3.5.1

USER root

RUN pip install \
    pyspark \
    xgboost \
    pandas \
    pyarrow \
    numpy \
    jupyter