import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLOps - Réadmission Diabète")
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["prepare", "train", "evaluate", "all"],
        help="Étape du pipeline à exécuter",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="diabetic_data.csv",
        help="Chemin vers le fichier CSV des données",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="diabetes_model.joblib",
        help="Chemin de sauvegarde/chargement du modèle",
    )

    args = parser.parse_args()

    if args.step == "prepare":
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        print(f"Données prêtes : X_train={X_train.shape}, X_test={X_test.shape}")

    elif args.step == "train":
        X_train, _, y_train, _ = prepare_data(args.data)
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)

    elif args.step == "evaluate":
        _, X_test, _, y_test = prepare_data(args.data)
        model = load_model(args.model_path)
        evaluate_model(model, X_test, y_test)

    elif args.step == "all":
        print("--- 1. Préparation des données ---")
        X_train, X_test, y_train, y_test = prepare_data(args.data)

        print("--- 2. Entraînement du modèle ---")
        model = train_model(X_train, y_train)
        save_model(model, args.model_path)

        print("--- 3. Évaluation du modèle ---")
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
