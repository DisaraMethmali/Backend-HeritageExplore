"""
Quick Deployment Script
Run this to deploy your model to HuggingFace
"""

from huggingface_deployment import deploy_to_huggingface

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================

ADAPTER_PATH = "./lora_adapter"  # Path to your LoRA adapter
REPO_NAME = "YOUR_USERNAME/sri-vijaya-rajasinha-chatbot"  # Your repo name
HF_TOKEN = "hf_xxxxxxxxxxxxx"  # Your HuggingFace token

# ============================================================================
# DEPLOY
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting deployment...")
    
    results = deploy_to_huggingface(
        adapter_path=ADAPTER_PATH,
        repo_name=REPO_NAME,
        hf_token=HF_TOKEN,
        create_space=True,
        create_api=True
    )
    
    print("\n✅ Deployment complete!")
    print(f"Results: {results}")