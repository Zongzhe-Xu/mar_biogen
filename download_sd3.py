from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="stabilityai/stable-diffusion-3-medium-diffusers",
    revision="ea42f8cef0f178587cf766dc8129abd379c90671",
    cache_dir="/projects/besp/BiosignalGen_zitao/"
)