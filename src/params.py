params = {
    # training
    "n_updates": 1000,
    "checkpoint_iter": 100,
    "n_eval_iters": 100,
    "learning_rate": 3e-4,
    # model
    "block_size": 64,
    "batch_size": 64,
    "n_embd": 64,
    "n_heads": 2,
    "n_layers": 2,
    # "loaded_model": "optimal.model",
    "loaded_model": None,
}
