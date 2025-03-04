from huggingface_hub import snapshot_download

# 定义要下载的模型列表
models = [
    {
        "repo_id": "prs-eth/marigold-depth-v1-0",
        "local_dir": "./checkpoints/marigold-depth-v1-0"
    },
    {
        "repo_id": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "local_dir": "./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K"
    },
    {
        "repo_id": "Johanan0528/DepthLab",
        "local_dir": "./checkpoints/DepthLab"
    }
]

# 依次下载每个模型
for model in models:
    print(f"Downloading {model['repo_id']}...")
    snapshot_download(
        repo_id=model["repo_id"], 
        local_dir=model["local_dir"],
        max_workers=6,     # 减少并行下载数
        resume_download=True  # 启用断点续传
    )
    print(f"Finished downloading {model['repo_id']}")