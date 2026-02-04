"""
下载 Chinese-TinyBERT-L4 模型
使用 ModelScope 镜像（国内稳定）
"""

import os
from pathlib import Path

save_path = "./models/chinese-tinybert-l4-uncased"
Path(save_path).mkdir(parents=True, exist_ok=True)

# 方法1: 使用 ModelScope (推荐)
try:
    print("尝试使用 ModelScope 下载...")
    from modelscope import snapshot_download
    
    model_dir = snapshot_download(
        'hf-models/chinese-tinybert-l4-uncased',
        cache_dir='./models',
        local_dir=save_path
    )
    print(f"ModelScope 下载成功: {model_dir}")
    
except Exception as e:
    print(f"ModelScope 下载失败: {e}")
    print("\n尝试使用 Hugging Face 镜像...")
    
    # 方法2: 使用 Hugging Face 镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        model_name = "hfl/chinese-tinybert-l4-uncased"
        
        print("下载配置...")
        config = AutoConfig.from_pretrained(model_name)
        config.save_pretrained(save_path)
        
        print("下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(save_path)
        
        print("下载模型权重...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(save_path)
        
        print(f"Hugging Face 下载成功！")
        
    except Exception as e2:
        print(f"Hugging Face 也失败了: {e2}")
        print("\n请手动下载：")
        print("1. 访问 https://hf-mirror.com/hfl/chinese-tinybert-l4-uncased")
        print("2. 点击 'Files and versions'")
        print("3. 下载所有文件到 ./models/chinese-tinybert-l4-uncased/")
        exit(1)

print(f"\n模型保存在: {Path(save_path).absolute()}")
print("训练时使用本地路径: ./models/chinese-tinybert-l4-uncased")
