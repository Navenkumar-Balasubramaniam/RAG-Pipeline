# Setup Guide

This project runs in a conda environment on **WSL Ubuntu** (or native Linux/macOS).
Native Windows is not supported because PaddlePaddle's CPU wheels are unreliable
on Windows and several other libraries play poorly with Windows file paths.

## Prerequisites

Install these on your system before setting up the project:

| Tool      | Purpose                          | Source                                     |
|-----------|----------------------------------|--------------------------------------------|
| Miniconda | Python environment management    | https://docs.conda.io/projects/miniconda/  |
| Git       | Version control                  | https://git-scm.com/downloads              |
| VS Code   | Editor                           | https://code.visualstudio.com/             |
| Ollama    | Local LLM runtime                | https://ollama.com/download                |

For WSL users, the VS Code **WSL extension** (by Microsoft) is required so VS
Code can run the Python interpreter inside WSL.

## One-time WSL system dependencies

PaddlePaddle's C++ runtime relies on `libgomp` (GNU OpenMP). Minimal Ubuntu
images (including default WSL Ubuntu) don't ship with it. Install once:

```bash
sudo apt update
sudo apt install -y libgomp1
```

Without this, `import paddle` fails with:
`ImportError: libgomp.so.1: cannot open shared object file: No such file or directory`

## Creating the conda environment

```bash
conda env create -f environment.yaml
conda activate rag-pipeline
```

To recreate after pulling environment.yaml changes:

```bash
conda env update -f environment.yaml --prune
```

To wipe and start fresh:

```bash
conda env remove -n rag-pipeline -y
conda env create -f environment.yaml
```

## Pulling the local LLM

This project runs Mistral 7B locally via Ollama. Pull the model once:

```bash
ollama pull mistral:7b-instruct
```

The model is ~4 GB and runs on CPU.

## Configuring VS Code

After creating the environment, tell VS Code to use it:

1. `Ctrl/Cmd+Shift+P` → **Python: Select Interpreter**
2. Pick the interpreter labeled `('rag-pipeline')`

If it isn't listed, click **Enter interpreter path...** and paste:
~/miniconda3/envs/rag-pipeline/bin/python

## Verifying the install

```bash
python -c "import paddle, paddleocr, llama_index.core; print('all good')"
```

If this prints `all good` without errors, the environment is ready.