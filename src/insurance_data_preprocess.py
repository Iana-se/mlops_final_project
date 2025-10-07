"""
Script: process_insurance_data.py
Description: Optimized PySpark script for processing Medical Cost Personal Datasets.
"""

from argparse import ArgumentParser
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

accesskey = "accesskey"
secretkey = "secretkey"
data_path = "s3a://data_path/"

def remove_outliers_iqr_spark(df, numeric_columns):
    """Удаление выбросов с использованием метода IQR в Spark"""
    for column in numeric_columns:
        quantiles = df.approxQuantile(column, [0.25, 0.75], 0.01)
        if not quantiles or len(quantiles) < 2:
            continue
        Q1, Q3 = quantiles
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df.filter((col(column) >= lower_bound) & (col(column) <= upper_bound))
    return df


def process_insurance_data(source_path: str, output_path: str):
    """
    PySpark pipeline:
    - Load data from S3
    - Clean and remove outliers
    - Encode categorical features
    - Scale numeric features
    - Split into train/test
    - Save results in Parquet format
    """
    spark = (
    SparkSession.builder
    .appName("TransactionValidation")
    # тюнинг Spark под ноды (s3-c4-m16)
    .config("spark.executor.instances", "3")   # по одному экзекутору на каждую ноду
    .config("spark.executor.cores", "4")       # используем все CPU
    .config("spark.executor.memory", "12g")    # оставляем запас на OS + YARN (из 16 ГБ)
    .config("spark.sql.shuffle.partitions", "300")  # меньше шардов → меньше shuffle
    .config("spark.memory.fraction", "0.6")    # больше памяти под dataframes, меньше под shuffle spill
    # S3 доступ
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    .config("spark.hadoop.fs.s3a.access.key", accesskey)
    .config("spark.hadoop.fs.s3a.secret.key", secretkey)
    .config("spark.hadoop.fs.s3a.endpoint", "storage.yandexcloud.net")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .getOrCreate())

    log = spark._jvm.org.apache.log4j.LogManager.getLogger(__name__)
    log.info("Starting insurance data processing...")

    # === Load Data ===
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(source_path)
    log.info(f"Loaded dataset with {len(df.columns)} columns.")

    # === Clean Data ===
    numeric_columns = ["age", "bmi", "children", "charges"]
    log.info("Removing outliers using IQR method...")
    df_clean = remove_outliers_iqr_spark(df, numeric_columns).dropna()
    log.info("Outlier removal and NA cleaning completed.")

    # === Split Data ===
    log.info("Splitting data into train/test sets...")
    train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

    # === Pipeline Setup ===
    categorical_cols = ["sex", "smoker", "region"]
    numeric_cols = ["age", "bmi", "children"]

    indexers = [
        StringIndexer(
            inputCol=c,
            outputCol=f"{c}_idx",
            handleInvalid="keep"
        ).setStringOrderType("alphabetAsc")
        for c in categorical_cols
    ]

    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical_cols]

    assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="numeric_scaled",
        withMean=True,
        withStd=True
    )

    feature_cols = [f"{c}_ohe" for c in categorical_cols] + ["numeric_scaled"]
    assembler_features = VectorAssembler(inputCols=feature_cols, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [assembler_numeric, scaler, assembler_features])

    # === Fit and Transform ===
    log.info("Fitting pipeline and transforming datasets...")
    pipeline_model = pipeline.fit(train_df)
    train_transformed = pipeline_model.transform(train_df)
    test_transformed = pipeline_model.transform(test_df)

    # === Save Results ===
    log.info("Saving transformed data and pipeline model...")
    train_transformed.select("features", "charges").write.mode("overwrite").parquet(f"{output_path}/train.parquet")
    test_transformed.select("features", "charges").write.mode("overwrite").parquet(f"{output_path}/test.parquet")
    
    log.info("Processing completed successfully.")
    spark.stop()


def main():
    parser = ArgumentParser()
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    args = parser.parse_args()
    bucket_name = args.bucket

    if not bucket_name:
        raise ValueError("Environment variable S3_BUCKET_NAME is not set")

    input_path = f"s3a://{bucket_name}/input_data/insurance.csv"
    output_path = f"s3a://{bucket_name}/output_data/"#processed_fraud_data.parquet"
    process_insurance_data(input_path, output_path)


if __name__ == "__main__":
    main()
