"""
Spark ML Pipeline 公共函数库
- 读取特征配置文件 (txt 格式)
- 创建 SparkSession
- 构建预处理 Pipeline
- 评估函数
"""

import os
import glob
import sys
import time
import json

# ── Spark 路径补丁 (apache/spark 镜像需要) ──
sys.path.insert(0, "/opt/spark/python")
for z in glob.glob("/opt/spark/python/lib/py4j-*.zip"):
    sys.path.insert(0, z)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    Imputer, StandardScaler,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import boto3


# ────────────────────────────────────────────────────────────────
# 配置读取
# ────────────────────────────────────────────────────────────────

def load_feature_config(base_dir: str = "."):
    """
    从 txt 文件读取特征配置。
    返回 (cat_features: list, num_features: list, label_col: str)
    """
    with open(os.path.join(base_dir, "cat_features.txt")) as f:
        cat_features = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    with open(os.path.join(base_dir, "num_features.txt")) as f:
        num_features = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    with open(os.path.join(base_dir, "label.txt")) as f:
        label_col = f.readline().strip()

    return cat_features, num_features, label_col


# ────────────────────────────────────────────────────────────────
# SparkSession
# ────────────────────────────────────────────────────────────────

def create_spark_session(app_name: str = "Spark-ML-Pipeline",
                         shuffle_partitions: int = 2) -> SparkSession:
    """创建连接 Spark 集群 + MinIO S3 的 SparkSession。"""
    return SparkSession.builder \
        .appName(app_name) \
        .master("spark://spark-master:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "1") \
        .config("spark.driver.memory", "2g") \
        .config("spark.task.cpus", "1") \
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
        .config("spark.default.parallelism", str(shuffle_partitions * 2)) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.hadoop.fs.s3a.fast.upload", "true") \
        .getOrCreate()


def get_s3_client():
    """获取 MinIO S3 客户端。"""
    return boto3.client(
        "s3", endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin", aws_secret_access_key="minioadmin",
        region_name="us-east-1",
    )


# ────────────────────────────────────────────────────────────────
# 数据加载
# ────────────────────────────────────────────────────────────────

def load_csv(spark: SparkSession, path: str, sep: str = ";",
             sample_size: int = None, seed: int = 42) -> DataFrame:
    """
    加载 CSV 文件，可选采样。
    自动重命名含 '.' 的列名。
    """
    df = spark.read.option("header", "true").option("sep", sep) \
        .option("inferSchema", "true").csv(path)

    # 重命名带点号的列名 (Spark ML 不支持)
    for c in df.columns:
        if "." in c:
            df = df.withColumnRenamed(c, c.replace(".", "_"))

    if sample_size is not None:
        total = df.count()
        ratio = sample_size / total
        df, _ = df.randomSplit([ratio, 1 - ratio], seed=seed)

    return df


def encode_label(df: DataFrame, label_col: str,
                 positive: str = "yes") -> DataFrame:
    """将字符串标签转换为 0/1。"""
    return df.withColumn(
        label_col,
        F.when(F.col(label_col) == positive, 1).otherwise(0).cast(DoubleType())
    )


def split_train_val_test(df: DataFrame, weights=(0.7, 0.15, 0.15),
                         seed: int = 42, n_partitions: int = 1):
    """拆分为训练/验证/测试集，返回 (train, val, test)。"""
    w0, w1, w2 = weights
    train, rest = df.randomSplit([w0, w1 + w2], seed=seed)
    val, test = rest.randomSplit([w1 / (w1 + w2), w2 / (w1 + w2)], seed=seed)
    train = train.repartition(n_partitions).cache()
    val = val.repartition(n_partitions).cache()
    test = test.repartition(n_partitions).cache()
    return train, val, test


# ────────────────────────────────────────────────────────────────
# Pipeline 构建
# ────────────────────────────────────────────────────────────────

def build_preprocessing_pipeline(cat_features: list, num_features: list) -> Pipeline:
    """
    构建预处理 Pipeline:
    - 类别特征: StringIndexer → OneHotEncoder
    - 数值特征: Imputer(median) → VectorAssembler → StandardScaler
    - 最终: 组装为 'features' 向量
    返回 Pipeline 和 one-hot 列名列表。
    """
    stages = []
    ohe_cols = []

    # 类别特征
    for feat in cat_features:
        idx_col = f"{feat}_idx"
        ohe_col = f"{feat}_ohe"
        ohe_cols.append(ohe_col)
        stages.append(StringIndexer(inputCol=feat, outputCol=idx_col, handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCol=idx_col, outputCol=ohe_col, handleInvalid="keep"))

    # 数值特征
    imp_cols = []
    for feat in num_features:
        imp_col = f"{feat}_imp"
        imp_cols.append(imp_col)
        stages.append(Imputer(inputCol=feat, outputCol=imp_col, strategy="median"))

    stages.append(VectorAssembler(inputCols=imp_cols, outputCol="num_raw", handleInvalid="keep"))
    stages.append(StandardScaler(inputCol="num_raw", outputCol="num_scaled", withStd=True, withMean=True))

    # 最终组装
    stages.append(VectorAssembler(inputCols=ohe_cols + ["num_scaled"], outputCol="features", handleInvalid="keep"))

    return Pipeline(stages=stages)


def transform_and_cache(pipeline_model: PipelineModel, df: DataFrame,
                        label_col: str, n_partitions: int = 1) -> DataFrame:
    """Transform 并缓存，返回 (features, label) 的 DataFrame。"""
    result = pipeline_model.transform(df).select("features", label_col)
    result = result.repartition(n_partitions).cache()
    result.count()  # 触发缓存
    return result


# ────────────────────────────────────────────────────────────────
# 评估
# ────────────────────────────────────────────────────────────────

def evaluate(predictions: DataFrame, label_col: str) -> dict:
    """计算分类指标。"""
    metrics = {}
    metrics["auc_roc"] = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    ).evaluate(predictions)
    metrics["auc_pr"] = BinaryClassificationEvaluator(
        labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    ).evaluate(predictions)
    metrics["accuracy"] = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="accuracy"
    ).evaluate(predictions)
    metrics["f1"] = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)
    metrics["precision"] = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="precisionByLabel"
    ).evaluate(predictions)
    metrics["recall"] = MulticlassClassificationEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="recallByLabel"
    ).evaluate(predictions)

    # 混淆矩阵
    tp = predictions.filter((F.col(label_col) == 1) & (F.col("prediction") == 1)).count()
    tn = predictions.filter((F.col(label_col) == 0) & (F.col("prediction") == 0)).count()
    fp = predictions.filter((F.col(label_col) == 0) & (F.col("prediction") == 1)).count()
    fn = predictions.filter((F.col(label_col) == 1) & (F.col("prediction") == 0)).count()
    metrics["tp"] = tp
    metrics["tn"] = tn
    metrics["fp"] = fp
    metrics["fn"] = fn
    return metrics


def print_metrics(metrics: dict, prefix: str = ""):
    """格式化打印指标。"""
    label = f"[{prefix}] " if prefix else ""
    print(f"{label}AUC(ROC)={metrics['auc_roc']:.4f}  AUC(PR)={metrics['auc_pr']:.4f}  "
          f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
          f"Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")
    print(f"{label}CM: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")


# ────────────────────────────────────────────────────────────────
# S3 保存 / 加载
# ────────────────────────────────────────────────────────────────

def save_pipeline(pm: PipelineModel, s3_path: str = "s3a://preprocessing/bank_marketing_pipeline"):
    """保存 Pipeline 到 S3。"""
    pm.write().overwrite().save(s3_path)
    print(f"Pipeline saved to {s3_path}")


def load_pipeline(s3_path: str = "s3a://preprocessing/bank_marketing_pipeline") -> PipelineModel:
    """从 S3 加载 Pipeline。"""
    return PipelineModel.load(s3_path)


def save_metadata(meta: dict, bucket: str = "preprocessing", key: str = "bank_meta.json"):
    """保存元数据 JSON 到 S3。"""
    s3 = get_s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(meta, indent=2).encode())
    print(f"Metadata saved to s3://{bucket}/{key}")


# ────────────────────────────────────────────────────────────────
# 特征重要性
# ────────────────────────────────────────────────────────────────

def print_feature_importance(model, top_n: int = 15):
    """打印 XGBoost 特征重要性 (by gain)。"""
    imp = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"Top {top_n} features by gain:")
    for feat, score in sorted_imp:
        print(f"  {feat}: {score:.1f}")
