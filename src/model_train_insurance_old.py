"""
Script: model_train_insurance.py
Description: PySpark regression training script for medical insurance cost prediction
             на основе уже подготовленного вектора features.
"""

import os
import sys
import numpy as np
import traceback
import argparse
import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient

from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder


def create_spark_session(s3_config=None):
    """
    Create and configure a Spark session.

    Parameters
    ----------
    s3_config : dict, optional
        Dictionary containing S3 configuration parameters
        (endpoint_url, access_key, secret_key)

    Returns
    -------
    SparkSession
        Configured Spark session
    """
    print("DEBUG: Начинаем создание Spark сессии")
    try:
        # Создаем базовый Builder
        builder = (SparkSession
            .builder
            .appName("FraudDetectionModel")
        )

        # Если передана конфигурация S3, добавляем настройки
        if s3_config and all(k in s3_config for k in ['endpoint_url', 'access_key', 'secret_key']):
            print(f"DEBUG: Настраиваем S3 с endpoint_url: {s3_config['endpoint_url']}")
            builder = (builder
                .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
                .config("spark.hadoop.fs.s3a.endpoint", s3_config['endpoint_url'])
                .config("spark.hadoop.fs.s3a.access.key", s3_config['access_key'])
                .config("spark.hadoop.fs.s3a.secret.key", s3_config['secret_key'])
                .config("spark.hadoop.fs.s3a.path.style.access", "true")
                .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "true")
            )

        # Настройки памяти и ресурсов Spark
        builder = (builder
            .config("spark.executor.memory", "8g")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.cores", "4")
            .config("spark.sql.shuffle.partitions", "200")
        )

        print("DEBUG: Spark сессия успешно сконфигурирована")
        # Создаем и возвращаем сессию Spark
        return builder
    except Exception as e:
        print(f"ERROR: Ошибка создания Spark сессии: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


def load_data(spark, input_path):
    """
    Load and prepare the fraud detection dataset.

    Parameters
    ----------
    spark : SparkSession
        Spark session
    input_path : str
        Path to the input data

    Returns
    -------
    tuple
        (train_df, test_df) - Spark DataFrames for training and testing
    """
    print(f"DEBUG: Начинаем загрузку данных из: {input_path}")
    try:
        # Load the data
        print(f"DEBUG: Чтение parquet данных из {input_path}")

        train_df = spark.read.parquet(f'{input_path}/train.parquet')
        test_df = spark.read.parquet(f'{input_path}/test.parquet')

        return train_df, test_df
    except Exception as e:
        print(f"ERROR: Ошибка загрузки данных: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

def train_model(
    train_df,
    test_df,
    label_col="charges",
    features_col="features",
    run_name="insurance_models",
    num_trees_grid=(50, 100),
    max_depth_grid=(5, 8),
    gbt_max_iter_grid=(50, 100),
    gbt_max_depth_grid=(3, 5),
    lr_reg_param_grid=(0.0, 0.1, 0.3) 
):
    """
    Train and compare RandomForest, GradientBoosting, and LinearRegression.
    Log only the best model to MLflow.
    """
    try:
        # === ЭВАЛЮАТОРЫ ===
        evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
        evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

        # ==========================================================
        # === RANDOM FOREST (МОДЕЛЬ №1) ============================
        # ==========================================================
        print("\n=== TRAINING RANDOM FOREST MODEL ===")
        rf = RandomForestRegressor(featuresCol=features_col, labelCol=label_col)
        rf_param_grid = (
            ParamGridBuilder()
            .addGrid(rf.numTrees, list(num_trees_grid))
            .addGrid(rf.maxDepth, list(max_depth_grid))
            .build()
        )
        rf_tvs = TrainValidationSplit(
            estimator=rf,
            estimatorParamMaps=rf_param_grid,
            evaluator=evaluator_rmse,
            trainRatio=0.8,
            parallelism=2
        )
        rf_model = rf_tvs.fit(train_df).bestModel
        rf_predictions = rf_model.transform(test_df)
        rf_rmse = evaluator_rmse.evaluate(rf_predictions)
        rf_r2 = evaluator_r2.evaluate(rf_predictions)
        print(f"RF -> RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")

        # ==========================================================
        # === GRADIENT BOOSTING (МОДЕЛЬ №2) ========================
        # ==========================================================
        print("\n=== TRAINING GRADIENT BOOSTING MODEL ===")
        gbt = GBTRegressor(featuresCol=features_col, labelCol=label_col, seed=42)
        gbt_param_grid = (
            ParamGridBuilder()
            .addGrid(gbt.maxIter, list(gbt_max_iter_grid))
            .addGrid(gbt.maxDepth, list(gbt_max_depth_grid))
            .build()
        )
        gbt_tvs = TrainValidationSplit(
            estimator=gbt,
            estimatorParamMaps=gbt_param_grid,
            evaluator=evaluator_rmse,
            trainRatio=0.8,
            parallelism=2
        )
        gbt_model = gbt_tvs.fit(train_df).bestModel
        gbt_predictions = gbt_model.transform(test_df)
        gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
        gbt_r2 = evaluator_r2.evaluate(gbt_predictions)
        print(f"GBT -> RMSE: {gbt_rmse:.4f}, R2: {gbt_r2:.4f}")

        # ==========================================================
        # === LINEAR REGRESSION (МОДЕЛЬ №3 - ДОБАВЛЕНА) ============
        # ==========================================================
        print("\n=== TRAINING LINEAR REGRESSION MODEL ===")
        lr = LinearRegression(featuresCol=features_col, labelCol=label_col)
        lr_param_grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, list(lr_reg_param_grid))
            .build()
        )
        lr_tvs = TrainValidationSplit(
            estimator=lr,
            estimatorParamMaps=lr_param_grid,
            evaluator=evaluator_rmse,
            trainRatio=0.8,
            parallelism=2
        )
        lr_model = lr_tvs.fit(train_df).bestModel
        lr_predictions = lr_model.transform(test_df)
        lr_rmse = evaluator_rmse.evaluate(lr_predictions)
        lr_r2 = evaluator_r2.evaluate(lr_predictions)
        print(f"LR -> RMSE: {lr_rmse:.4f}, R2: {lr_r2:.4f}")

        # ==========================================================
        # === СРАВНЕНИЕ ВСЕХ 3 МОДЕЛЕЙ =============================
        # ==========================================================
        models = {
            "RandomForestRegressor": (rf_model, rf_rmse, rf_r2),
            "GBTRegressor": (gbt_model, gbt_rmse, gbt_r2),
            "LinearRegression": (lr_model, lr_rmse, lr_r2),
        }

        best_name, (best_model, best_rmse, best_r2) = min(models.items(), key=lambda x: x[1][1])
        print(f"\n=== BEST MODEL: {best_name} (RMSE={best_rmse:.4f}, R2={best_r2:.4f}) ===")

        # ==========================================================
        # === ЛОГИРОВАНИЕ В MLflow =================================
        # ==========================================================
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            mlflow.log_param("best_model", best_name)
            mlflow.log_metric("rmse", float(best_rmse))
            mlflow.log_metric("r2", float(best_r2))
            mlflow.spark.log_model(best_model, "model")
            print(f"MLflow Run ID: {run_id} - Logged {best_name}")

            return best_model, {"run_id": run_id, "rmse": best_rmse, "r2": best_r2, "model": best_name}

    except Exception as e:
        mlflow.log_text(f"Traceback: {traceback.format_exc()}", "exception_while_training.txt")
        raise

# def train_model(
#     train_df,
#     test_df,
#     label_col="charges",
#     features_col="features",
#     run_name="insurance_models",
#     num_trees_grid=(50, 100),
#     max_depth_grid=(5, 8),
#     gbt_max_iter_grid=(50, 100),     # === ДОБАВЛЕНО: сетка для GBT ===
#     gbt_max_depth_grid=(3, 5),       # === ДОБАВЛЕНО: сетка для GBT ===
# ):
#     """
#     Train and compare RandomForestRegressor and GradientBoostingRegressor on preprocessed data.
#     Log only the best performing model to MLflow.
#     """
#     try:
#         # === ОБЩИЕ ЭВАЛЮАТОРЫ ===
#         evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
#         evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

#         # ==========================================================
#         # === RANDOM FOREST MODEL TRAINING BLOCK ===================
#         # ==========================================================
#         rf = RandomForestRegressor(featuresCol=features_col, labelCol=label_col)

#         rf_param_grid = (
#             ParamGridBuilder()
#             .addGrid(rf.numTrees, list(num_trees_grid))
#             .addGrid(rf.maxDepth, list(max_depth_grid))
#             .build()
#         )

#         rf_tvs = TrainValidationSplit(
#             estimator=rf,
#             estimatorParamMaps=rf_param_grid,
#             evaluator=evaluator_rmse,
#             trainRatio=0.8,
#             parallelism=2
#         )

#         print("\n=== TRAINING RANDOM FOREST MODEL ===")
#         rf_model = rf_tvs.fit(train_df).bestModel

#         rf_predictions = rf_model.transform(test_df)
#         rf_rmse = evaluator_rmse.evaluate(rf_predictions)
#         rf_r2 = evaluator_r2.evaluate(rf_predictions)

#         print(f"RF -> RMSE: {rf_rmse:.4f}, R2: {rf_r2:.4f}")

#         # ==========================================================
#         # === GRADIENT BOOSTING MODEL TRAINING BLOCK ===============
#         # ==========================================================
#         print("\n=== TRAINING GRADIENT BOOSTING MODEL ===")
#         gbt = GBTRegressor(featuresCol=features_col, labelCol=label_col, seed=42)

#         gbt_param_grid = (
#             ParamGridBuilder()
#             .addGrid(gbt.maxIter, list(gbt_max_iter_grid))
#             .addGrid(gbt.maxDepth, list(gbt_max_depth_grid))
#             .build()
#         )

#         gbt_tvs = TrainValidationSplit(
#             estimator=gbt,
#             estimatorParamMaps=gbt_param_grid,
#             evaluator=evaluator_rmse,
#             trainRatio=0.8,
#             parallelism=2
#         )

#         gbt_model = gbt_tvs.fit(train_df).bestModel

#         gbt_predictions = gbt_model.transform(test_df)
#         gbt_rmse = evaluator_rmse.evaluate(gbt_predictions)
#         gbt_r2 = evaluator_r2.evaluate(gbt_predictions)

#         print(f"GBT -> RMSE: {gbt_rmse:.4f}, R2: {gbt_r2:.4f}")

#         # ==========================================================
#         # === СРАВНЕНИЕ МОДЕЛЕЙ И ВЫБОР ЛУЧШЕЙ ===================
#         # ==========================================================
#         if gbt_rmse < rf_rmse:
#             best_model = gbt_model
#             best_metrics = {"rmse": gbt_rmse, "r2": gbt_r2, "model": "GBTRegressor"}
#         else:
#             best_model = rf_model
#             best_metrics = {"rmse": rf_rmse, "r2": rf_r2, "model": "RandomForestRegressor"}

#         print(f"\n=== BEST MODEL: {best_metrics['model']} (RMSE={best_metrics['rmse']:.4f}) ===")

#         # ==========================================================
#         # === ЛОГИРОВАНИЕ В MLflow ТОЛЬКО ЛУЧШЕЙ МОДЕЛИ ===========
#         # ==========================================================
#         with mlflow.start_run(run_name=run_name) as run:
#             run_id = run.info.run_id

#             mlflow.log_param("best_model", best_metrics["model"])
#             mlflow.log_metric("rmse", float(best_metrics["rmse"]))
#             mlflow.log_metric("r2", float(best_metrics["r2"]))
#             mlflow.spark.log_model(best_model, "model")

#             print(f"MLflow Run ID: {run_id} - Logged {best_metrics['model']}")

#             return best_model, {"run_id": run_id, **best_metrics}

#     except Exception as e:
#         mlflow.log_text(f"Traceback: {traceback.format_exc()}", "exception_while_training.txt")
#         raise


# def train_model(
#     train_df,
#     test_df,
#     label_col="charges",
#     features_col="features",
#     run_name="insurance_rf",
#     num_trees_grid=(50, 100),
#     max_depth_grid=(5, 8),
# ):
#     """
#     Train RandomForestRegressor on preprocessed data (features vector already present).

#     Args:
#         train_df (DataFrame): training Spark DataFrame (must contain `features` and `charges` by default)
#         test_df (DataFrame): testing Spark DataFrame
#         label_col (str): name of label column (default "charges")
#         features_col (str): name of features vector column (default "features")
#         run_name (str): MLflow run name
#         num_trees_grid (iterable): values for numTrees grid
#         max_depth_grid (iterable): values for maxDepth grid

#     Returns:
#         best_model (PipelineModel / RandomForestRegressionModel): trained model
#         dict: metrics {"run_id", "rmse", "r2"}
#     """
#     try:
#         # Создаём регрессор, предполагая, что признаки уже собраны в features_col
#         rf = RandomForestRegressor(featuresCol=features_col, labelCol=label_col)

#         # Параметрическая сетка
#         param_grid = (
#             ParamGridBuilder()
#             .addGrid(rf.numTrees, list(num_trees_grid))
#             .addGrid(rf.maxDepth, list(max_depth_grid))
#             .build()
#         )

#         # Оценщики для регрессии
#         evaluator_rmse = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
#         evaluator_r2 = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="r2")

#         # TrainValidationSplit
#         tvs = TrainValidationSplit(
#             estimator=rf,
#             estimatorParamMaps=param_grid,
#             evaluator=evaluator_rmse,
#             trainRatio=0.8,
#             parallelism=2
#         )

#         # MLflow run
#         with mlflow.start_run(run_name=run_name) as run:
#             run_id = run.info.run_id

#             # Обучаем модель на train_df
#             tvs_model = tvs.fit(train_df)
#             best_model = tvs_model.bestModel  # RandomForestRegressionModel

#             # Получение лучших параметров RandomForest
#             # rf_model = best_model.stages[-1]  # последняя стадия Pipeline
#             # best_num_trees = rf_model.getNumTrees
#             # best_max_depth = rf_model.getMaxDepth()
#             best_num_trees = best_model.getNumTrees
#             best_max_depth = best_model.getMaxDepth()


#             # Логирование лучших параметров
#             mlflow.log_param("best_numTrees", best_num_trees)
#             mlflow.log_param("best_maxDepth", best_max_depth)

#             # Предсказания и метрики на тесте
#             predictions = best_model.transform(test_df)
#             rmse = evaluator_rmse.evaluate(predictions)
#             r2 = evaluator_r2.evaluate(predictions)

#             # Логируем метрики и модель в MLflow
#             mlflow.log_metric("rmse", float(rmse))
#             mlflow.log_metric("r2", float(r2))
#             mlflow.spark.log_model(best_model, "model")

#             print(f"MLflow Run ID: {run_id} - RMSE: {rmse:.4f}, R2: {r2:.4f}")

#             return best_model, {"run_id": run_id, "rmse": rmse, "r2": r2}

#     except Exception as e:
#         mlflow.log_text(f"Traceback: {traceback.format_exc()}", "exception_while_training.txt")
#         raise

def save_model(model, output_path):
    """
    Save the trained model to the specified path.

    Parameters
    ----------
    model : PipelineModel
        Trained model
    output_path : str
        Path to save the model
    """
    print(f"DEBUG: Сохраняем модель в: {output_path}")
    try:
        model.write().overwrite().save(output_path)
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Ошибка сохранения модели: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

# --------------------------------------------------------
# MLflow utilities
# --------------------------------------------------------
def get_best_model_metrics(experiment_name):
    """
    ПОЛУЧАЕМ МЕТРИКИ ЛУЧШЕЙ РЕГРЕССИОННОЙ МОДЕЛИ ИЗ MLflow С АЛИАСОМ 'CHAMPION'

    Parameters
    ----------
    experiment_name : str
        Имя эксперимента MLflow

    Returns
    -------
    dict
        Метрики лучшей модели или None, если модели нет
    """
    print(f"DEBUG: Получаем метрики лучшей модели для эксперимента '{experiment_name}'")
    client = MlflowClient()

    try:
        print(f"DEBUG: Ищем эксперимент {experiment_name}")
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"Эксперимент '{experiment_name}' не найден")
            return None
        print(f"DEBUG: Эксперимент найден, ID: {experiment.experiment_id}")
    except Exception as e:
        print(f"ERROR: Ошибка при получении эксперимента: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

    try:
        model_name = f"{experiment_name}_model"
        print(f"DEBUG: Ищем зарегистрированную модель '{model_name}'")

        try:
            registered_model = client.get_registered_model(model_name)
            print(f"Модель '{model_name}' зарегистрирована")
            print(f"Модель '{model_name}' имеет {len(registered_model.latest_versions)} версий")
        except Exception as e:
            print(f"DEBUG: Модель '{model_name}' еще не зарегистрирована: {str(e)}")
            return None

        model_versions = client.get_latest_versions(model_name)
        champion_version = None

        print(f"DEBUG: Найдено {len(model_versions)} версий модели")
        for version in model_versions:
            print(f"DEBUG: Проверяем версию {version.version}")
            if hasattr(version, 'aliases') and "champion" in version.aliases:
                print(f"DEBUG: Найден 'champion' в aliases: {version.aliases}")
                champion_version = version
                break
            elif hasattr(version, 'tags') and version.tags.get('alias') == "champion":
                print(f"DEBUG: Найден 'champion' в тегах: {version.tags}")
                champion_version = version
                break
            else:
                print(f"DEBUG: Версия {version.version} не имеет алиаса 'champion'")
                if hasattr(version, 'aliases'):
                    print(f"DEBUG: Aliases: {version.aliases}")
                if hasattr(version, 'tags'):
                    print(f"DEBUG: Tags: {version.tags}")

        if not champion_version:
            print("Модель с алиасом 'champion' не найдена")
            return None

        champion_run_id = champion_version.run_id
        print(f"DEBUG: Run ID для 'champion': {champion_run_id}")

        # 🔹 РЕГРЕССИЯ: меняем метрики на rmse и r2
        print(f"DEBUG: Получаем метрики для run_id: {champion_run_id}")
        run = client.get_run(champion_run_id)
        metrics = {
            "run_id": champion_run_id,
            "rmse": run.data.metrics.get("rmse"),
            "r2": run.data.metrics.get("r2")
        }

        print(
            f"Текущая лучшая модель (champion): "
            f"версия {champion_version.version}, Run ID: {champion_run_id}"
        )
        print(
            f"Метрики: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}"
        )

        return metrics
    except Exception as e:
        print(f"ERROR: Ошибка при получении лучшей модели: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None


def compare_and_register_model(new_metrics, experiment_name):
    """
    СРАВНИВАЕМ НОВУЮ РЕГРЕССИОННУЮ МОДЕЛЬ С ЛУЧШЕЙ В MLflow И РЕГИСТРИРУЕМ, ЕСЛИ ОНА ЛУЧШЕ

    Parameters
    ----------
    new_metrics : dict
        Метрики новой модели (должны содержать 'rmse' и 'r2')
    experiment_name : str
        Имя эксперимента MLflow

    Returns
    -------
    bool
        True, если новая модель была зарегистрирована как 'champion'
    """
    print(f"DEBUG: Сравниваем и регистрируем модель для эксперимента {experiment_name}")
    client = MlflowClient()

    # Получаем метрики лучшей модели
    print("DEBUG: Получаем метрики лучшей модели")
    best_metrics = get_best_model_metrics(experiment_name)

    # Имя модели
    model_name = f"{experiment_name}_model"
    print(f"DEBUG: Имя модели: {model_name}")

    # Создаем или получаем регистрированную модель
    try:
        print(f"DEBUG: Проверяем существует ли модель {model_name}")
        client.get_registered_model(model_name)
        print(f"Модель '{model_name}' уже зарегистрирована")
    except Exception as e:
        print(f"DEBUG: Создаем новую модель: {str(e)}")
        client.create_registered_model(model_name)
        print(f"Создана новая регистрированная модель '{model_name}'")

    # Регистрируем новую модель как новую версию
    run_id = new_metrics["run_id"]
    model_uri = f"runs:/{run_id}/model"
    print(f"DEBUG: Регистрируем модель из {model_uri}")
    model_details = mlflow.register_model(model_uri, model_name)
    new_version = model_details.version
    print(f"DEBUG: Зарегистрирована новая версия: {new_version}")

    # РЕШЕНИЕ, ДОЛЖНА ЛИ МОДЕЛЬ СТАТЬ CHAMPION
    should_promote = False

    if not best_metrics:
        should_promote = True
        print("ЭТО ПЕРВАЯ РЕГИСТРИРУЕМАЯ МОДЕЛЬ, ОНА СТАНОВИТСЯ 'CHAMPION'")
    else:
        # 🔹 СРАВНИВАЕМ ПО RMSE (ЧЕМ МЕНЬШЕ, ТЕМ ЛУЧШЕ)
        print(f"DEBUG: Сравниваем метрики - текущий RMSE: {best_metrics['rmse']}, новый RMSE: {new_metrics['rmse']}")
        if new_metrics["rmse"] < best_metrics["rmse"]:
            should_promote = True
            improvement = (best_metrics["rmse"] - new_metrics["rmse"]) / best_metrics["rmse"] * 100
            print(f"Новая модель лучше на {improvement:.2f}% по RMSE. Установка в качестве 'champion'")
        else:
            print(f"Новая модель хуже текущей 'champion' по RMSE ({new_metrics['rmse']:.4f} >= {best_metrics['rmse']:.4f})")

    # УСТАНАВЛИВАЕМ АЛИАС 'CHAMPION' ДЛЯ ЛУЧШЕЙ МОДЕЛИ
    if should_promote:
        try:
            print("DEBUG: Пытаемся установить алиас 'champion'")
            if hasattr(client, 'set_registered_model_alias'):
                print("DEBUG: Используем set_registered_model_alias")
                client.set_registered_model_alias(model_name, "champion", new_version)
            else:
                print("DEBUG: Используем set_model_version_tag")
                client.set_model_version_tag(model_name, new_version, "alias", "champion")
        except Exception as e:
            print(f"ERROR: Ошибка установки алиаса 'champion': {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            print("DEBUG: Используем set_model_version_tag (запасной вариант)")
            client.set_model_version_tag(model_name, new_version, "alias", "champion")

        print(f"Версия {new_version} модели '{model_name}' установлена как 'champion'")
        return True

    # Если модель не лучше, устанавливаем алиас 'challenger'
    try:
        print("DEBUG: Пытаемся установить алиас 'challenger'")
        if hasattr(client, 'set_registered_model_alias'):
            client.set_registered_model_alias(model_name, "challenger", new_version)
        else:
            client.set_model_version_tag(model_name, new_version, "alias", "challenger")
    except Exception as e:
        print(f"ERROR: Ошибка установки алиаса 'challenger': {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        print("DEBUG: Используем set_model_version_tag (запасной вариант)")
        client.set_model_version_tag(model_name, new_version, "alias", "challenger")

    print(f"Версия {new_version} модели '{model_name}' установлена как 'challenger'")
    return False

def main():
    print("DEBUG: Скрипт запущен, начинаем инициализацию")
    parser = argparse.ArgumentParser(description="Insurance Model Training")
    # Основные параметры
    parser.add_argument("--input", required=True, help="Input data path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--model-type", default="various", help="Model type (rf or lr)")

    # MLflow параметры
    parser.add_argument("--tracking-uri", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="Insurance", help="MLflow exp name")
    parser.add_argument("--auto-register", action="store_true", help="Automatically register")
    parser.add_argument("--run-name", default=None, help="Name for the MLflow run")

    # Отключение проверки Git для MLflow
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

    # S3 параметры
    parser.add_argument("--s3-endpoint-url", help="S3 endpoint URL")
    parser.add_argument("--s3-access-key", help="S3 access key")
    parser.add_argument("--s3-secret-key", help="S3 secret key")

    args = parser.parse_args()
    print(f"DEBUG: Аргументы командной строки: {args}")

    # Настраиваем S3 конфигурацию
    s3_config = None
    if args.s3_endpoint_url and args.s3_access_key and args.s3_secret_key:
        print("DEBUG: Настраиваем S3 конфигурацию")
        s3_config = {
            'endpoint_url': args.s3_endpoint_url,
            'access_key': args.s3_access_key,
            'secret_key': args.s3_secret_key
        }
        # Устанавливаем переменные окружения для MLflow
        os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.s3_secret_key
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url
        print("DEBUG: Переменные окружения для S3 установлены")

    # Set MLflow tracking URI if provided
    if args.tracking_uri:
        print(f"DEBUG: Устанавливаем MLflow tracking URI: {args.tracking_uri}")
        mlflow.set_tracking_uri(args.tracking_uri)

    # Create or set the experiment
    print(f"DEBUG: Устанавливаем MLflow эксперимент: {args.experiment_name}")
    mlflow.set_experiment(args.experiment_name)

    # Create Spark session
    print("DEBUG: Создаем Spark сессию")
    spark = create_spark_session(s3_config).getOrCreate()
    print("DEBUG: Spark сессия создана")

    try:
        train_df, test_df = load_data(spark, args.input)
        # Generate run name if not provided
        run_name = args.run_name or f"insurance_{args.model_type}_{os.path.basename(args.input)}"
        print(f"DEBUG: Run name: {run_name}")

        model, metrics = train_model(train_df, test_df, run_name=run_name)

        # Save the model locally
        print("DEBUG: Сохраняем модель")
        save_model(model, args.output)

        # Register model if requested
        if args.auto_register:
            print("DEBUG: Сравниваем и регистрируем модель")
            compare_and_register_model(metrics, args.experiment_name)

        print("Training completed successfully!")

    except Exception as e:
        print(f"ERROR: Ошибка во время обучения: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Stop Spark session
        print("DEBUG: Останавливаем Spark сессию")
        spark.stop()
        print("DEBUG: Скрипт завершен")


if __name__ == "__main__":
    main()

