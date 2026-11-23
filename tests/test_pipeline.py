def test_model_pipeline_exports():
    import model_pipeline

    for fn in ("prepare_data", "train_model", "evaluate_model", "save_model", "load_model"):
        assert hasattr(model_pipeline, fn)
