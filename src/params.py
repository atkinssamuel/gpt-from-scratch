params = {
    # training
    "n_updates": 2500,
    "checkpoint_iter": 50,
    "n_eval_iters": 5,
    "learning_rate": 3e-4,
    # model
    "block_size": 256,
    "batch_size": 64,
    "n_embd": 384,
    "n_heads": 6,
    "n_layers": 6,
    "loaded_model_name": None,
    "saved_model_name": "optimal.model",
}
