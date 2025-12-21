from huggingface_hub import hf_hub_download

# Download the file
file_path = hf_hub_download(
    repo_id="kevinhuangxf/MarkSplatter",
    filename="marksplatter.safetensors",
    repo_type="model",
    local_dir="./ckpts"  # Change to your desired path, or remove to download to cache
)
print(f"Downloaded to: {file_path}")

file_path = hf_hub_download(
    repo_id="kevinhuangxf/MarkSplatter",
    filename="gaussianbridge.safetensors",
    repo_type="model",
    local_dir="./ckpts"  # Change to your desired path, or remove to download to cache
)
print(f"Downloaded to: {file_path}")
