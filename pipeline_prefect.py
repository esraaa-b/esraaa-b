import argparse
import subprocess
import sys
from prefect import flow, get_run_logger, task
from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DEFAULT_DATA = "diabetic_data.csv"
DEFAULT_MODEL_PATH = "diabetes_model.joblib"
TARGET_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


# --- Tasks ---
@task(name="install-dependencies")
def task_install_dependencies(requirements_path: str = "requirements.txt") -> None:
    logger = get_run_logger()
    logger.info(f"Installation des dépendances depuis {requirements_path}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True
    )


@task(name="format-code")
def task_format_code() -> None:
    logger = get_run_logger()
    logger.info(f"Formatage du code sur : {', '.join(TARGET_FILES)}")
    subprocess.run([sys.executable, "-m", "black", *TARGET_FILES], check=True)


@task(name="code-quality")
def task_code_quality() -> None:
    logger = get_run_logger()
    logger.info(f"Qualité du code sur : {', '.join(TARGET_FILES)}")
    subprocess.run(
        [sys.executable, "-m", "flake8", "--max-line-length=88", *TARGET_FILES],
        check=True,
    )


@task(name="code-security")
def task_code_security() -> None:
    logger = get_run_logger()
    logger.info(f"Sécurité du code sur : {', '.join(TARGET_FILES)}")
    subprocess.run([sys.executable, "-m", "bandit", "-ll", *TARGET_FILES], check=True)


@task(name="unit-tests")
def task_unit_tests() -> None:
    logger = get_run_logger()
    logger.info("Exécution des tests unitaires...")
    subprocess.run([sys.executable, "-m", "pytest", "tests"], check=True)


@task(name="prepare-data")
def task_prepare_data(data_path: str = DEFAULT_DATA):
    logger = get_run_logger()
    logger.info(f"Préparation des données depuis {data_path}")
    return prepare_data(data_path)


@task(name="train-model")
def task_train_model(X_train, y_train):
    logger = get_run_logger()
    logger.info("Entraînement du modèle...")
    return train_model(X_train, y_train)


@task(name="save-model")
def task_save_model(model, model_path: str = DEFAULT_MODEL_PATH) -> None:
    logger = get_run_logger()
    logger.info(f"Sauvegarde du modèle vers {model_path}")
    save_model(model, model_path)


@task(name="load-model")
def task_load_model(model_path: str = DEFAULT_MODEL_PATH):
    logger = get_run_logger()
    logger.info(f"Chargement du modèle depuis {model_path}")
    return load_model(model_path)


@task(name="evaluate-model")
def task_evaluate_model(model, X_test, y_test):
    logger = get_run_logger()
    logger.info("Évaluation du modèle")
    return evaluate_model(model, X_test, y_test)


# --- Flows ---
@flow(name="code", log_prints=True)
def flow_code() -> None:
    task_format_code()
    task_code_quality()
    task_code_security()
    task_unit_tests()


@flow(name="train", log_prints=True)
def flow_train(data_path: str = DEFAULT_DATA, model_path: str = DEFAULT_MODEL_PATH):
    X_train, _, y_train, _ = task_prepare_data(data_path)
    model = task_train_model(X_train, y_train)
    task_save_model(model, model_path)
    return model


@flow(name="evaluate", log_prints=True)
def flow_evaluate(data_path: str = DEFAULT_DATA, model_path: str = DEFAULT_MODEL_PATH):
    _, X_test, _, y_test = task_prepare_data(data_path)
    model = task_load_model(model_path)
    return task_evaluate_model(model, X_test, y_test)


@flow(name="all", log_prints=True)
def flow_all(data_path: str = DEFAULT_DATA, model_path: str = DEFAULT_MODEL_PATH):
    flow_code()
    X_train, X_test, y_train, y_test = task_prepare_data(data_path)
    model = task_train_model(X_train, y_train)
    task_save_model(model, model_path)
    return task_evaluate_model(model, X_test, y_test)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline Prefect - Réadmission Diabète"
    )
    parser.add_argument(
        "--flow",
        type=str,
        required=True,
        choices=["all", "train", "entrainement", "evaluate", "code"],
        help="Flow Prefect à exécuter",
    )
    parser.add_argument(
        "--data", type=str, default=DEFAULT_DATA, help="Chemin du fichier CSV"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Chemin du modèle",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.flow == "all":
        flow_all(data_path=args.data, model_path=args.model_path)
    elif args.flow in ("train", "entrainement"):
        flow_train(data_path=args.data, model_path=args.model_path)
    elif args.flow == "evaluate":
        flow_evaluate(data_path=args.data, model_path=args.model_path)
    elif args.flow == "code":
        flow_code()
