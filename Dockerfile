FROM apache/spark:3.5.1

USER root

# S3A 相关 JAR - 使用阿里云 Maven 镜像加速
ARG HADOOP_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.367

RUN cd /opt/spark/jars && \
    curl -sO "https://maven.aliyun.com/repository/public/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar" && \
    curl -sO "https://maven.aliyun.com/repository/public/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar"

# Python 依赖 - 使用清华源加速
# apache/spark 镜像已有 pyspark/numpy/pandas，只需额外安装
ENV PYTHONPATH="/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip:${PYTHONPATH}"

RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    pandas \
    numpy \
    pyarrow \
    xgboost \
    scikit-learn \
    jupyter \
    jupyterlab-lsp \
    python-lsp-server \
    boto3 \
    s3fs \
    pyyaml