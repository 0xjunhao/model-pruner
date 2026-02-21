import argparse
from huggingface_hub import HfApi, hf_hub_download
import json
import os
from safetensors.torch import load_file, save_file
import tempfile
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import import_utils

GiB = 1024 ** 3
MAX_SHARD_SIZE = 4 * GiB


def shim_transformer5x() -> None:
    # `is_torch_fx_available` was removed in transformers 5.x.
    # We add a shim to avoid errors for transformers 4.x based models.
    if not hasattr(import_utils, "is_torch_fx_available"):
        def is_torch_fx_available():
            try:
                import torch.fx
                return True
            except ImportError:
                return False
        import_utils.is_torch_fx_available = is_torch_fx_available


def create_readme(args: argparse.Namespace, output_dir: str) -> None:
    print(f"Generating README.md for the pruned model: {args.target}")
    try:
        readme_path = hf_hub_download(
            repo_id=args.source, filename="README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            if content.startswith("---"):
                content = content.split("---", 2)[-1].strip()
    except Exception as e:
        print(f"Failed to download README.md from source model: {e}")
        content = ""
    config = AutoConfig.from_pretrained(args.source, trust_remote_code=True)
    source_layers = config.num_hidden_layers
    content = f"""---
base_model: {args.source}
library: transformers
tags:
- pruned
---

*This model is a pruned variant of {args.source} that retains the first 
{args.layers} layer(s) of the original {source_layers} layer(s) architecture.
It is intended for performance research rather than production use.*

""" + content
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(content)


def download_and_consolidate_weights(
        args: argparse.Namespace, output_dir: str) -> None:
    # Obtain relevant weight names
    config = AutoConfig.from_pretrained(args.source, trust_remote_code=True)
    source_layers = config.num_hidden_layers
    print(f"Source model layers: {source_layers}")
    if source_layers <= args.layers:
        print("No pruning needed.")
        exit(0)
    config.num_hidden_layers = args.layers
    for key in ["layer_types"]:
        if hasattr(config, key):
            current_list = getattr(config, key)
            if isinstance(current_list, list):
                setattr(config, key, current_list[:args.layers])
    config.save_pretrained(output_dir)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True)
    relevant_weight_names = set(model.state_dict().keys())

    # Obtain relevant shards
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files(repo_id=args.source)
    if "model.safetensors.index.json" in repo_files:
        index_path = hf_hub_download(
            repo_id=args.source, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            source_index = json.load(f)
        source_weight_map = source_index["weight_map"]
        relevant_source_shards = {
            source_weight_map[name]
            for name in relevant_weight_names if name in source_weight_map}
        relevant_source_shards = sorted(list(relevant_source_shards))
    elif "model.safetensors" in repo_files:
        relevant_source_shards = ["model.safetensors"]
    else:
        print("No model.safetensors or model.safetensors.index.json "
              "found in the source model repository.")
        exit(1)
    print(f"Relevant source shards: {relevant_source_shards}")

    # Download and consolidate relevant shards
    target_weight_map = {}
    target_shard_count = 1
    buffer_size = 0
    total_size = 0
    buffer_dict = {}
    target_shard = f"model-{target_shard_count:05d}.safetensors"
    for source_shard in tqdm(relevant_source_shards, desc="Processing shards"):
        shard_path = hf_hub_download(
            repo_id=args.source, filename=source_shard)
        source_weights = load_file(shard_path)
        for weight_name, weight in source_weights.items():
            if not weight_name in relevant_weight_names:
                continue
            weight_size = weight.numel() * weight.element_size()
            buffer_size += weight_size
            total_size += weight_size
            buffer_dict[weight_name] = weight
            target_weight_map[weight_name] = target_shard
            if buffer_size > MAX_SHARD_SIZE:
                save_file(buffer_dict, os.path.join(output_dir, target_shard))
                target_shard_count += 1
                target_shard = f"model-{target_shard_count:05d}.safetensors"
                buffer_dict, buffer_size = {}, 0
    if buffer_size > 0:
        save_file(buffer_dict, os.path.join(output_dir, target_shard))
    with open(os.path.join(
            output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({
            "metadata": {"total_size": total_size},
            "weight_map": target_weight_map
        }, f, indent=2)


def main(args: argparse.Namespace):
    print(f"Source model: {args.source}")
    print(f"Target model: {args.target}")
    print(f"Number of layers to keep: {args.layers}")

    base_cache_dir = os.path.expanduser("~/.cache/model_pruner")
    output_dir = os.path.join(base_cache_dir, args.target.replace('/', '--'))
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    shim_transformer5x()
    download_and_consolidate_weights(args, output_dir)
    for extra_file in ["LICENSE", "vocab.json", "merges.txt",
                       "tokenizer_config.json", "tokenizer.json",
                       "generation_config.json"]:
        try:
            hf_hub_download(repo_id=args.source,
                            filename=extra_file, local_dir=output_dir)
        except Exception:
            pass
    create_readme(args, output_dir)
    print(f"Pruning complete. You can find the pruned model at: {output_dir}")

    # Upload to Hugging Face Hub
    if not args.upload:
        return
    print(f"Uploading to Hugging Face Hub: {args.target}")
    hf_api = HfApi()
    hf_api.create_repo(repo_id=args.target, repo_type="model", exist_ok=True)
    hf_api.upload_folder(
        folder_path=output_dir,
        repo_id=args.target,
        repo_type="model",
        commit_message=f"Pruned to {args.layers} layers"
    )
    print(f"Model has been uploaded to: https://huggingface.co/{args.target}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Ubicloud Model Pruner")
    argparser.add_argument("--source", type=str, required=True,
                           help="The source model to be pruned. "
                           "E.g. 'deepseek-ai/DeepSeek-R1'")
    argparser.add_argument("--target", type=str, required=True,
                           help="The target model after pruning. "
                           "E.g. 'ubicloud/DeepSeek-R1-Pruned'")
    argparser.add_argument("--layers", type=int, default=8,
                           help="The number of layers to keep. E.g. 8")
    argparser.add_argument("--upload", action="store_true",
                           help="Whether to upload to Hugging Face.")
    args = argparser.parse_args()

    main(args)
