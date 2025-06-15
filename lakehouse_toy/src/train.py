from pyspark.ml import Pipeline
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import logging
from pathlib import Path
from typing import Tuple, List
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(spark: SparkSession, gold_path: str) -> Tuple:
    logger.info(f"Loading data from gold layer: {gold_path}")

    if not os.path.exists(gold_path):
        raise ValueError(f"Path does not exist: {gold_path}")

    df = spark.read.format("delta").load(gold_path)

    all_cols = [c for c in df.columns if c not in ["Credit_Score"]]

    numeric_cols = [
        c for c in all_cols if c.endswith(("_q1", "_median", "_q3", "_avg", "_std"))
    ]
    categorical_cols = [c for c in all_cols if c.endswith("_encoded")]

    logger.info(f"Selected numeric features: {numeric_cols}")
    logger.info(f"Selected categorical features: {categorical_cols}")

    assembler = VectorAssembler(
        inputCols=numeric_cols + categorical_cols, outputCol="features"
    )

    df = assembler.transform(df)

    indexer = StringIndexer(
        inputCol="Credit_Score", outputCol="label", handleInvalid="keep"
    )
    df = indexer.fit(df).transform(df)

    label_dist = df.groupBy("label").count().orderBy("label")
    logger.info("Label distribution:")
    label_dist.show()

    final_count = df.count()
    logger.info(f"Final row count: {final_count}")

    if final_count == 0:
        raise ValueError("No valid data rows after preprocessing")

    return df, numeric_cols + categorical_cols


def train_model(df: DataFrame, feature_cols: List[str]):
    logger.info("Starting training...")

    df = df.cache()

    model = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
    )

    pipeline = Pipeline(stages=[model])

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    paramMap = {
        model.numTrees: 75,
        model.maxDepth: 5,
        model.minInstancesPerNode: 5,
        model.seed: 42,
    }

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=[paramMap],
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
    )

    cv_model = cv.fit(df)

    avg_metrics = cv_model.avgMetrics
    logger.info(f"Average F1 score across folds: {avg_metrics}")

    df.unpersist()

    return cv_model.bestModel


def calculate_metrics(model, df):
    logger.info("Calculating evaluation metrics...")

    predictions = model.transform(df)

    pred_pd = predictions.select("label", "prediction", "probability").toPandas()

    metrics = {}

    metrics["f1_macro"] = f1_score(
        pred_pd["label"], pred_pd["prediction"], average="macro"
    )

    metrics["auc_roc"] = roc_auc_score(
        pred_pd["label"],
        pred_pd["probability"].apply(lambda x: x.toArray()),
        multi_class="ovr",
    )

    metrics["avg_preecision"] = average_precision_score(
        pred_pd["label"],
        pred_pd["probability"].apply(lambda x: x.toArray()),
        average="macro",
    )

    return metrics


def save_model(model, output_path):
    logger.info(f"Saving model to {output_path}")
    model_path = Path(output_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(model_path))
