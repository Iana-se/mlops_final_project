"""
Script: model_train_insurance.py
Description: PySpark regression training script for medical insurance cost prediction
             с A/B тестированием против текущего чемпиона в MLflow
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
from scipy import stats
import warnings

warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))


def create_spark_session(s3_config=None):
    """
    Create and configure a Spark session.
    """
    print("DEBUG: Начинаем создание Spark сессии")
    try:
        builder = (SparkSession
            .builder
            .appName("FraudDetectionModel")
        )

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

        builder = (builder
            .config("spark.executor.memory", "8g")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.cores", "4")
            .config("spark.sql.shuffle.partitions", "200")
        )

        print("DEBUG: Spark сессия успешно сконфигурирована")
        return builder
    except Exception as e:
        print(f"ERROR: Ошибка создания Spark сессии: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


def load_data(spark, input_path):
    """
    Load and prepare the fraud detection dataset.
    """
    print(f"DEBUG: Начинаем загрузку данных из: {input_path}")
    try:
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
        # === LINEAR REGRESSION (МОДЕЛЬ №3) ========================
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


def bootstrap_metrics(predictions, evaluator_rmse, evaluator_r2, n_iterations=100):
    """
    Выполняет bootstrap-анализ метрик для регрессии.
    
    Parameters
    ----------
    predictions : DataFrame
        Предсказания модели
    evaluator_rmse : RegressionEvaluator
        Эвалюатор для RMSE
    evaluator_r2 : RegressionEvaluator
        Эвалюатор для R2
    n_iterations : int
        Количество итераций bootstrap
        
    Returns
    -------
    dict
        Словарь с bootstrap распределениями метрик
    """
    print(f"DEBUG: Bootstrap анализ ({n_iterations} итераций)")
    
    # Конвертируем в pandas для удобства bootstrap
    pred_pd = predictions.select("prediction", "charges").toPandas()
    
    rmse_scores = []
    r2_scores = []
    
    np.random.seed(42)
    
    for i in range(n_iterations):
        # Bootstrap sample
        sample = pred_pd.sample(frac=1.0, replace=True)
        
        # Вычисляем метрики
        rmse = np.sqrt(np.mean((sample['charges'] - sample['prediction'])**2))
        r2 = 1 - np.sum((sample['charges'] - sample['prediction'])**2) / np.sum((sample['charges'] - sample['charges'].mean())**2)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    return {
        "rmse": np.array(rmse_scores),
        "r2": np.array(r2_scores)
    }


def statistical_comparison(prod_bootstrap, cand_bootstrap, alpha=0.05):
    """
    Статистическое сравнение двух моделей с помощью t-теста.
    
    Parameters
    ----------
    prod_bootstrap : dict
        Bootstrap метрики production модели
    cand_bootstrap : dict
        Bootstrap метрики candidate модели
    alpha : float
        Уровень значимости
        
    Returns
    -------
    dict
        Результаты статистического сравнения
    """
    print(f"DEBUG: Статистическое сравнение (α={alpha})")
    
    results = {}
    
    for metric in ['rmse', 'r2']:
        # T-тест для разницы метрик
        t_stat, p_value = stats.ttest_ind(prod_bootstrap[metric], cand_bootstrap[metric])
        
        # Размер эффекта (Cohen's d)
        pooled_std = np.sqrt((np.var(prod_bootstrap[metric]) + np.var(cand_bootstrap[metric])) / 2)
        effect_size = abs(np.mean(cand_bootstrap[metric]) - np.mean(prod_bootstrap[metric])) / pooled_std
        
        # Для RMSE улучшение = уменьшение, для R2 улучшение = увеличение
        if metric == 'rmse':
            improvement = np.mean(prod_bootstrap[metric]) - np.mean(cand_bootstrap[metric])  # положительное = улучшение
            is_improvement = improvement > 0
        else:  # r2
            improvement = np.mean(cand_bootstrap[metric]) - np.mean(prod_bootstrap[metric])  # положительное = улучшение
            is_improvement = improvement > 0
        
        is_significant = p_value < alpha
        
        results[metric] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'is_significant': is_significant,
            'is_improvement': is_improvement,
            'production_mean': np.mean(prod_bootstrap[metric]),
            'candidate_mean': np.mean(cand_bootstrap[metric]),
            'improvement': improvement,
            'improvement_percent': (improvement / np.mean(prod_bootstrap[metric])) * 100
        }
    
    return results


def ab_test_models(production_model, candidate_model, test_df, alpha=0.05):
    """
    A/B ТЕСТИРОВАНИЕ: сравнение production и candidate моделей.
    
    Parameters
    ----------
    production_model : Model
        Текущая production модель (champion)
    candidate_model : Model
        Новая candidate модель
    test_df : DataFrame
        Тестовые данные
    alpha : float
        Уровень значимости
        
    Returns
    -------
    dict
        Результаты A/B теста
    """
    print("\n" + "="*60)
    print("A/B ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
    print("="*60)
    
    # Эвалюаторы
    evaluator_rmse = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="charges", predictionCol="prediction", metricName="r2")
    
    # Предсказания production модели
    print("Оценка PRODUCTION модели...")
    prod_predictions = production_model.transform(test_df)
    prod_rmse = evaluator_rmse.evaluate(prod_predictions)
    prod_r2 = evaluator_r2.evaluate(prod_predictions)
    
    print(f"PRODUCTION -> RMSE: {prod_rmse:.4f}, R2: {prod_r2:.4f}")
    
    # Предсказания candidate модели
    print("Оценка CANDIDATE модели...")
    cand_predictions = candidate_model.transform(test_df)
    cand_rmse = evaluator_rmse.evaluate(cand_predictions)
    cand_r2 = evaluator_r2.evaluate(cand_predictions)
    
    print(f"CANDIDATE -> RMSE: {cand_rmse:.4f}, R2: {cand_r2:.4f}")
    
    # Bootstrap анализ
    print("Bootstrap анализ PRODUCTION модели...")
    prod_bootstrap = bootstrap_metrics(prod_predictions, evaluator_rmse, evaluator_r2)
    
    print("Bootstrap анализ CANDIDATE модели...")
    cand_bootstrap = bootstrap_metrics(cand_predictions, evaluator_rmse, evaluator_r2)
    
    # Статистическое сравнение
    comparison_results = statistical_comparison(prod_bootstrap, cand_bootstrap, alpha)
    
    # Вывод результатов
    print("\n" + "="*50)
    print("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО СРАВНЕНИЯ")
    print("="*50)
    
    for metric, results in comparison_results.items():
        print(f"\n{metric.upper()}:")
        print(f"  Production: {results['production_mean']:.4f}")
        print(f"  Candidate:  {results['candidate_mean']:.4f}")
        print(f"  Улучшение:  {results['improvement']:+.4f} ({results['improvement_percent']:+.2f}%)")
        print(f"  p-value:    {results['p_value']:.6f}")
        print(f"  Cohen's d:  {results['effect_size']:.4f}")
        
        if results['is_significant'] and results['is_improvement']:
            print(f"  ✅ ЗНАЧИМОЕ УЛУЧШЕНИЕ при α={alpha}")
        elif results['is_significant'] and not results['is_improvement']:
            print(f"  ❌ ЗНАЧИМОЕ УХУДШЕНИЕ при α={alpha}")
        else:
            print(f"  ⚠️  Незначимое различие при α={alpha}")
    
    # Общее решение на основе RMSE (основная метрика для регрессии)
    rmse_results = comparison_results['rmse']
    should_deploy = rmse_results['is_significant'] and rmse_results['is_improvement']
    
    print("\n" + "="*50)
    print("ИТОГОВОЕ РЕШЕНИЕ:")
    if should_deploy:
        print("✅ РАЗВЕРНУТЬ новую модель в PRODUCTION")
        print("   Модель-кандидат показала статистически значимое улучшение RMSE")
    else:
        print("❌ ОСТАВИТЬ текущую модель в PRODUCTION")
        if not rmse_results['is_improvement']:
            print("   Модель-кандидат не показала улучшения RMSE")
        else:
            print("   Улучшение статистически незначимо")
    print("="*50)
    
    return {
        "should_deploy": should_deploy,
        "production_metrics": {"rmse": prod_rmse, "r2": prod_r2},
        "candidate_metrics": {"rmse": cand_rmse, "r2": cand_r2},
        "comparison_results": comparison_results,
        "production_bootstrap": prod_bootstrap,
        "candidate_bootstrap": cand_bootstrap
    }


def get_champion_model(experiment_name):
    """
    ПОЛУЧАЕМ ТЕКУЩУЮ PRODUCTION МОДЕЛЬ (CHAMPION) ИЗ MLflow
    
    Returns
    -------
    tuple : (model, metrics) или (None, None) если модель не найдена
    """
    print(f"DEBUG: Поиск champion модели для эксперимента '{experiment_name}'")
    client = MlflowClient()
    
    try:
        model_name = f"{experiment_name}_model"
        
        print(f"DEBUG: Ищем зарегистрированную модель '{model_name}'")
        
        try:
            registered_model = client.get_registered_model(model_name)
            print(f"Модель '{model_name}' зарегистрирована")
        except Exception as e:
            print(f"DEBUG: Модель '{model_name}' не найдена: {str(e)}")
            return None, None
        
        # Ищем версию с алиасом 'champion'
        model_versions = client.get_latest_versions(model_name)
        champion_version = None
        
        for version in model_versions:
            if hasattr(version, 'aliases') and "champion" in version.aliases:
                champion_version = version
                break
            elif hasattr(version, 'tags') and version.tags.get('alias') == "champion":
                champion_version = version
                break
        
        if not champion_version:
            print("Модель с алиасом 'champion' не найдена")
            return None, None
        
        champion_run_id = champion_version.run_id
        print(f"DEBUG: Run ID для 'champion': {champion_run_id}")
        
        # Загружаем модель
        model_uri = f"runs:/{champion_run_id}/model"
        champion_model = mlflow.spark.load_model(model_uri)
        
        # Получаем метрики
        run = client.get_run(champion_run_id)
        metrics = {
            "run_id": champion_run_id,
            "rmse": run.data.metrics.get("rmse"),
            "r2": run.data.metrics.get("r2"),
            "model_type": run.data.params.get("best_model", "unknown")
        }
        
        print(f"Champion модель найдена: {metrics['model_type']}")
        print(f"Метрики champion: RMSE={metrics['rmse']:.4f}, R2={metrics['r2']:.4f}")
        
        return champion_model, metrics
        
    except Exception as e:
        print(f"ERROR: Ошибка при загрузке champion модели: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None, None


def register_model_with_alias(model_name, run_id, alias, description=""):
    """
    РЕГИСТРИРУЕМ МОДЕЛЬ И УСТАНАВЛИВАЕМ АЛИАС
    """
    print(f"DEBUG: Регистрируем модель {model_name} с алиасом '{alias}'")
    client = MlflowClient()
    
    try:
        # Создаем или получаем зарегистрированную модель
        try:
            client.get_registered_model(model_name)
        except Exception:
            client.create_registered_model(model_name)
        
        # Регистрируем новую версию
        model_uri = f"runs:/{run_id}/model"
        model_details = mlflow.register_model(model_uri, model_name)
        new_version = model_details.version
        
        # Устанавливаем алиас
        try:
            if hasattr(client, 'set_registered_model_alias'):
                client.set_registered_model_alias(model_name, alias, new_version)
            else:
                client.set_model_version_tag(model_name, new_version, "alias", alias)
        except Exception as e:
            print(f"WARNING: Ошибка установки алиаса: {str(e)}")
            client.set_model_version_tag(model_name, new_version, "alias", alias)
        
        if description:
            client.update_model_version(
                name=model_name,
                version=new_version,
                description=description
            )
        
        print(f"Модель {model_name} версии {new_version} установлена как '{alias}'")
        return new_version
        
    except Exception as e:
        print(f"ERROR: Ошибка регистрации модели: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


def save_model(model, output_path):
    """
    Save the trained model to the specified path.
    """
    print(f"DEBUG: Сохраняем модель в: {output_path}")
    try:
        model.write().overwrite().save(output_path)
        print(f"Model saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Ошибка сохранения модели: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise


def main():
    print("DEBUG: Скрипт запущен, начинаем инициализацию")
    parser = argparse.ArgumentParser(description="Insurance Model Training with A/B Testing")
    
    # Основные параметры
    parser.add_argument("--input", required=True, help="Input data path")
    parser.add_argument("--output", required=True, help="Output model path")
    parser.add_argument("--model-type", default="various", help="Model type")
    
    # MLflow параметры
    parser.add_argument("--tracking-uri", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="Insurance", help="MLflow exp name")
    parser.add_argument("--auto-register", action="store_true", help="Automatically register")
    parser.add_argument("--run-name", default=None, help="Name for the MLflow run")
    parser.add_argument("--ab-test-alpha", type=float, default=0.05, help="A/B test significance level")
    
    # S3 параметры
    parser.add_argument("--s3-endpoint-url", help="S3 endpoint URL")
    parser.add_argument("--s3-access-key", help="S3 access key")
    parser.add_argument("--s3-secret-key", help="S3 secret key")
    
    args = parser.parse_args()
    print(f"DEBUG: Аргументы командной строки: {args}")

    # Отключение проверки Git для MLflow
    os.environ['GIT_PYTHON_REFRESH'] = 'quiet'

    # Настраиваем S3 конфигурацию
    s3_config = None
    if args.s3_endpoint_url and args.s3_access_key and args.s3_secret_key:
        print("DEBUG: Настраиваем S3 конфигурацию")
        s3_config = {
            'endpoint_url': args.s3_endpoint_url,
            'access_key': args.s3_access_key,
            'secret_key': args.s3_secret_key
        }
        os.environ['AWS_ACCESS_KEY_ID'] = args.s3_access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = args.s3_secret_key
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = args.s3_endpoint_url

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
        # Загружаем данные
        train_df, test_df = load_data(spark, args.input)
        
        # Проверяем есть ли champion модель в MLflow
        champion_model, champion_metrics = get_champion_model(args.experiment_name)
        
        if champion_model is None:
            print("\n" + "="*50)
            print("CHAMPION МОДЕЛЬ НЕ НАЙДЕНА")
            print("ОБУЧАЕМ И РЕГИСТРИРУЕМ ПЕРВУЮ БАЗОВУЮ МОДЕЛЬ")
            print("="*50)
            
            # Обучаем модель
            run_name = args.run_name or f"insurance_initial_{os.path.basename(args.input)}"
            model, metrics = train_model(train_df, test_df, run_name=run_name)
            
            # Сохраняем модель локально
            save_model(model, args.output)
            
            # Регистрируем как champion
            if args.auto_register:
                model_name = f"{args.experiment_name}_model"
                register_model_with_alias(
                    model_name=model_name,
                    run_id=metrics["run_id"],
                    alias="champion",
                )
            
        else:
            print("\n" + "="*50)
            print("CHAMPION МОДЕЛЬ НАЙДЕНА - ЗАПУСКАЕМ A/B ТЕСТ")
            print("="*50)
            
            # Обучаем новые модели
            run_name = args.run_name or f"insurance_candidate_{os.path.basename(args.input)}"
            candidate_model, candidate_metrics = train_model(train_df, test_df, run_name=run_name)
            
            # Сохраняем модель локально
            save_model(candidate_model, args.output)
            
            # A/B ТЕСТИРОВАНИЕ
            ab_results = ab_test_models(
                production_model=champion_model,
                candidate_model=candidate_model,
                test_df=test_df,
                alpha=args.ab_test_alpha
            )
            
            # Регистрируем модель в зависимости от результатов A/B теста
            if args.auto_register:
                model_name = f"{args.experiment_name}_model"
                
                if ab_results["should_deploy"]:
                    # Регистрируем как нового champion
                    new_version = register_model_with_alias(
                        model_name=model_name,
                        run_id=candidate_metrics["run_id"],
                        alias="champion",
                        description=f"Новый champion после A/B теста. Улучшение RMSE: {ab_results['comparison_results']['rmse']['improvement']:+.4f}"
                    )
                    print(f"✅ Новая модель версии {new_version} установлена как champion!")
                else:
                    # Регистрируем как challenger
                    new_version = register_model_with_alias(
                        model_name=model_name,
                        run_id=candidate_metrics["run_id"],
                        alias="challenger",
                        description=f"Challenger модель. RMSE: {candidate_metrics['rmse']:.4f}"
                    )
                    print(f"⚠️ Модель версии {new_version} установлена как challenger")

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