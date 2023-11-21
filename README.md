# GPT from Scratch
This repository contains a GPT-like model based on the [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial video.


# Environment Configuration
Ensure the [tatooine](https://github.com/atkinssamuel/tatooine) repository exists a directory above this one. 

```
├── gpt-from-scratch/
│   └── ...
└── tatooine/
    └── ...
```

Then, create a Python environment and install the dependencies in the `requirements.txt` file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```