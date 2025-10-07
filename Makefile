SHELL := /bin/bash

include .env

.EXPORT_ALL_VARIABLES:

.PHONY: generate-env
generate-env:
	@echo "Generating .env from infra/variables.json..."
	@python3 create_env.py
	@echo ".env файл успешно создан"

.PHONY: build-venv
build-venv:
	@echo "Building venv archive..."
	@bash ./scripts/create_venv_archive.sh
	@echo "Done: venvs/venv.tar.gz created"

.PHONY: upload-venv-to-bucket
upload-venv-to-bucket:
	@echo "Uploading virtual environment archive to $(S3_BUCKET_NAME)..."
	s3cmd put venvs/venv.tar.gz s3://$(S3_BUCKET_NAME)/venvs/venv.tar.gz
	@echo "Virtual environment archive uploaded successfully"

.PHONY: upload-dags-to-bucket
upload-dags-to-bucket:
	@echo "Uploading dags to $(S3_BUCKET_NAME)..."
	s3cmd put --recursive dags/ s3://$(S3_BUCKET_NAME)/dags/
	@echo "DAGs uploaded successfully"

.PHONY: upload-src-to-bucket
upload-src-to-bucket:
	@echo "Uploading src to $(S3_BUCKET_NAME)..."
	s3cmd put --recursive src/ s3://$(S3_BUCKET_NAME)/src/
	@echo "Src uploaded successfully"

.PHONY: upload-medical-data-to-bucket
upload-medical-data-to-bucket:
	@echo "Uploading data to $(S3_BUCKET_NAME)..."
	s3cmd put --recursive data/medical_data/*.csv s3://$(S3_BUCKET_NAME)/input_data/
	@echo "Data uploaded successfully"


airflow-medical_task_preprocess: generate-env upload-medical-data-to-bucket upload-venv-to-bucket upload-src-to-bucket upload-dags-to-bucket 