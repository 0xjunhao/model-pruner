# ✂️ Model Pruner

**Create lightweight versions of massive LLMs by truncating their transformer layers.**

This tool allows you to take a large model from the Hugging Face Hub (e.g., `DeepSeek-R1`), slice it down to the first $N$ layers, and push the resulting "mini-model" back to your own Hugging Face repository.

> **⚠️ Note:** This tool performs *structural pruning* (truncation). A model with only its first 2 layers will likely output gibberish. This tool is intended for infrastructure and pipeline testing, not for improving inference quality.


## 🚀 Why use this?

Loading a 700B+ parameter model just to test your inference pipeline or prototype a performance optimization is overkill. This tool creates **structurally identical but drastically smaller** models that fit into memory on far fewer and much smller GPUs, so that you can reduce development costs and iterate faster.

## 💻 Usage

First, obtain a Hugging Face *Write Token* so you can upload the pruned model.
You can generate one at [Hugging Face](https://huggingface.co/) and set it as an environment variable:

```bash
export HF_TOKEN="hf_..."
```

Next, install the dependencies and run the script, specifying the source model, target model name, and the number of layers to keep:

```bash
uv run python3 main.py --source deepseek-ai/DeepSeek-R1 --target ubicloud/DeepSeek-R1-Pruned-108B --layers 12 [--upload]
```

Sample output: [ubicloud/DeepSeek-R1-Pruned-108B](https://huggingface.co/ubicloud/DeepSeek-R1-Pruned-108B)


> **🚀 Tip:** This tool is designed to handle models far larger than your available system RAM (for example, processing a 700B-parameter model on a laptop with only 16 GB of memory).
It only downloads the relevant weights and processes them in a streaming fashion.
