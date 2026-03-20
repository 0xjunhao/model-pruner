import argparse
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import json
import os
import re
from safetensors.torch import load_file, save_file
from tqdm import tqdm

# Constants for chunking the output weights
GiB = 1024 ** 3
MAX_SHARD_SIZE = 4 * GiB


def download_config(model: str) -> dict:
    """
    Downloads and parses the config.json file for a given model from the Hugging Face Hub.
    """
    try:
        config_path = hf_hub_download(repo_id=model, filename="config.json")
    except Exception as e:
        print(f"Failed to download config.json from source model: {e}")
        exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def create_readme(args: argparse.Namespace, output_dir: str) -> None:
    """
    Generates a new README.md for the pruned model.
    It fetches the source model's README, strips any existing YAML frontmatter, 
    and prepends a new frontmatter and disclaimer about the pruning process.
    """
    print(f"Generating README.md for the pruned model: {args.target}")
    try:
        # Attempt to get the original README
        readme_path = hf_hub_download(
            repo_id=args.source, filename="README.md")
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Strip original YAML frontmatter if it exists
            if content.startswith("---"):
                content = content.split("---", 2)[-1].strip()
    except Exception as e:
        print(f"Failed to download README.md from source model: {e}")
        content = ""

    config = download_config(args.source)
    # Handle multimodal architectures that nest text config
    text_config = config.get("text_config", config)
    source_layers = text_config["num_hidden_layers"]

    # Construct the new README content with appropriate metadata
    content = f"""---
base_model: {args.source}
library: transformers
tags:
- pruned
---

*This model is a pruned variant of {args.source} that retains the first 
{args.layers} layer(s) of the original {source_layers} layer(s) architecture.
It is intended for pipeline testing and performance research rather than 
production use.*

Made with ❤️ by [Model Pruner](https://github.com/ubicloud/model-pruner.git)

""" + content

    # Save the new README to the output directory
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(content)


def should_keep(weight_name: str, layers_to_keep: int) -> bool:
    """
    Determines if a specific weight tensor should be kept based on its name.
    Extracts the layer index using regex and drops it if it exceeds our target.
    """
    # Look for patterns like '.layers.5.' or '.layers.12.' in the tensor name
    layer_id = re.search(r"\.layers\.(\d+)\.", weight_name)
    if layer_id is not None and int(layer_id.group(1)) >= layers_to_keep:
        return False
    # Keep non-layer specific weights (like embeddings, lm_head) and layers within our threshold
    return True


def download_and_consolidate_weights(
        args: argparse.Namespace, output_dir: str) -> None:
    """
    Downloads the necessary safetensors shards, filters out dropped layers,
    and repacks the remaining weights into new, consolidated shard files.
    """
    # Obtain relevant weight names and update configuration
    config = download_config(args.source)
    text_config = config.get("text_config", config)
    source_layers = text_config["num_hidden_layers"]
    print(f"Source model layers: {source_layers}")

    if source_layers <= args.layers:
        print(
            "No pruning needed. The source model has equal or fewer layers than requested.")
        exit(0)

    # Update the config to reflect the new number of layers
    text_config["num_hidden_layers"] = args.layers

    # Update layer types list if the architecture explicitly defines it
    for key in ["layer_types"]:
        if key in text_config:
            current_list = text_config[key]
            if isinstance(current_list, list):
                text_config[key] = current_list[:args.layers]

    # Save the updated configuration
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Obtain relevant shards by checking the safetensors index
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files(repo_id=args.source)

    if "model.safetensors.index.json" in repo_files:
        # Sharded model
        index_path = hf_hub_download(
            repo_id=args.source, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            source_index = json.load(f)

        source_weight_map = source_index["weight_map"]
        # Figure out which shard files actually contain weights we want to keep
        relevant_source_shards = {
            source_weight_map[name]
            for name in source_weight_map
            if should_keep(name, args.layers)
        }
        relevant_source_shards = sorted(list(relevant_source_shards))
    elif "model.safetensors" in repo_files:
        # Single file model
        relevant_source_shards = ["model.safetensors"]
    else:
        print("No model.safetensors or model.safetensors.index.json "
              "found in the source model repository.")
        exit(1)

    print(f"Downloading relevant source shards: {relevant_source_shards}")

    # Initialize variables for the new consolidated shards
    target_weight_map = {}
    target_shard_count = 1
    buffer_size = 0
    total_size = 0
    buffer_dict = {}
    target_shard = f"model-{target_shard_count:05d}.safetensors"

    # Pre-download all relevant source shards so we don't block during processing
    for source_shard in relevant_source_shards:
        hf_hub_download(repo_id=args.source, filename=source_shard)

    # Process and consolidate the weights
    for source_shard in tqdm(relevant_source_shards, desc="Consolidating relevant weights"):
        shard_path = hf_hub_download(
            repo_id=args.source, filename=source_shard)
        source_weights = load_file(shard_path)

        for weight_name, weight in source_weights.items():
            if not should_keep(weight_name, args.layers):
                continue

            # Calculate memory size of this tensor
            weight_size = weight.numel() * weight.element_size()
            buffer_size += weight_size
            total_size += weight_size
            buffer_dict[weight_name] = weight
            target_weight_map[weight_name] = target_shard

            # If the current buffer exceeds our max shard size (e.g., 4GB), save it and start a new shard
            if buffer_size > MAX_SHARD_SIZE:
                save_file(buffer_dict, os.path.join(output_dir, target_shard))
                target_shard_count += 1
                target_shard = f"model-{target_shard_count:05d}.safetensors"
                buffer_dict, buffer_size = {}, 0

    # Save any remaining weights in the final shard
    if buffer_size > 0:
        save_file(buffer_dict, os.path.join(output_dir, target_shard))

    # Create the new safetensors index file mapping weights to their new shards
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

    # Set up local caching/output directory
    base_cache_dir = os.path.expanduser("~/.cache/model_pruner")
    output_dir = os.path.join(base_cache_dir, args.target.replace('/', '--'))
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Download auxiliary files (tokenizers, generation configs, etc.)
    # We ignore the heavy weight files because we handle them separately.
    try:
        snapshot_download(repo_id=args.source, local_dir=output_dir, ignore_patterns=[
                          "*.safetensors", "model.safetensors.index.json"])
    except Exception:
        print("Failed to download non-weight files from source model.")
        exit(1)

    # 2. Process and consolidate the weights
    download_and_consolidate_weights(args, output_dir)

    # 3. Create the updated README
    create_readme(args, output_dir)
    print(f"Pruning complete. You can find the pruned model at: {output_dir}")

    # 4. Upload to Hugging Face Hub (Optional)
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
