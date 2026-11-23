from pipeline_prefect import (
    DEFAULT_DATA,
    DEFAULT_MODEL_PATH,
    flow_all,
    flow_code,
    flow_evaluate,
    flow_train,
)


def main():
    deploy_code = flow_code.to_deployment(
        name="code-deployment",
        cron="0 * * * *",
    )
    deploy_train = flow_train.to_deployment(
        name="train-deployment",
        parameters={"data_path": DEFAULT_DATA, "model_path": DEFAULT_MODEL_PATH},
        cron="0 2 * * *",
    )
    deploy_evaluate = flow_evaluate.to_deployment(
        name="evaluate-deployment",
        parameters={"data_path": DEFAULT_DATA, "model_path": DEFAULT_MODEL_PATH},
        cron="0 3 * * *",
    )
    deploy_all = flow_all.to_deployment(
        name="all-deployment",
        parameters={"data_path": DEFAULT_DATA, "model_path": DEFAULT_MODEL_PATH},
        cron="0 4 * * 0",
    )

    from prefect import serve

    serve(deploy_code, deploy_train, deploy_evaluate, deploy_all)


if __name__ == "__main__":
    main()
