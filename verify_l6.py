"""
验证 L6 模型能否正常加载
"""
import sys
sys.path.insert(0, '.')

print("="*60)
print("验证 L6 模型")
print("="*60)

# 检查文件
from pathlib import Path
model_dir = Path('models/chinese-tinybert-l6-uncased')
required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']

print("\n1. 检查文件完整性...")
for f in required_files:
    file_path = model_dir / f
    if file_path.exists():
        size = file_path.stat().st_size
        print(f"   ✓ {f} ({size/1024/1024:.1f} MB)")
    else:
        print(f"   ✗ {f} 缺失！")
        sys.exit(1)

print("\n2. 测试加载模型...")
try:
    from src.model import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name='./models/chinese-tinybert-l6-uncased',
        num_labels=18
    )
    print(f"   ✓ 模型加载成功")
    print(f"   ✓ 层数: {model.bert.config.num_hidden_layers}")
    print(f"   ✓ 隐藏层维度: {model.bert.config.hidden_size}")
    print(f"   ✓ 词表大小: {len(tokenizer)}")
except Exception as e:
    print(f"   ✗ 加载失败: {e}")
    sys.exit(1)

print("\n3. 测试前向传播...")
try:
    import torch
    test_input = tokenizer("测试文本", return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**test_input)
    print(f"   ✓ 前向传播成功")
    print(f"   ✓ 输出形状: {outputs['logits'].shape}")
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ L6 模型验证通过！可以开始训练")
print("="*60)
