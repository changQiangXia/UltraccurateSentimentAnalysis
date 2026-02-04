"""
PyTorch 模型导出为 ONNX 格式
支持动态 batch size 和序列长度
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.onnx

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SentimentAnalyzer
from transformers import AutoTokenizer


def export_to_onnx(
    model_path: str,
    output_path: str,
    max_length: int = 128,
    opset_version: int = 14
):
    """
    将 PyTorch 模型导出为 ONNX 格式
    
    Args:
        model_path: PyTorch 模型路径
        output_path: ONNX 输出路径
        max_length: 最大序列长度
        opset_version: ONNX opset 版本
    """
    print(f"正在加载模型：{model_path}")
    
    # 加载配置
    config_path = Path(model_path) / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载模型
    model = SentimentAnalyzer(
        model_name=config['model_name'],
        num_labels=config['num_labels'],
        dropout=config.get('hidden_dropout_prob', 0.1)
    )
    
    state_dict = torch.load(Path(model_path) / 'pytorch_model.bin', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 创建示例输入
    sample_text = "这是一个测试文本"
    inputs = tokenizer(
        sample_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids))
    
    print(f"输入形状：input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    
    # 定义动态轴
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'token_type_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
    
    # 导出 ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"正在导出 ONNX 模型到：{output_path}")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask, token_type_ids),
            str(output_path),
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['logits'],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True
        )
    
    print(f"ONNX 模型导出成功！")
    
    # 验证 ONNX 模型
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print(f"ONNX 模型验证通过！")
        
        # 打印模型信息
        print(f"\n模型信息：")
        print(f"  IR 版本：{onnx_model.ir_version}")
        print(f"  Opset 版本：{onnx_model.opset_import[0].version}")
        
        # 估算模型大小
        model_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  模型大小：{model_size:.2f} MB")
        
    except ImportError:
        print("未安装 onnx 包，跳过验证")
    except Exception as e:
        print(f"ONNX 验证失败：{e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='导出 ONNX 模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='PyTorch 模型路径')
    parser.add_argument('--output_path', type=str, default='deployments/model.onnx',
                        help='ONNX 输出路径')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--opset_version', type=int, default=14,
                        help='ONNX opset 版本')
    
    args = parser.parse_args()
    
    export_to_onnx(
        model_path=args.model_path,
        output_path=args.output_path,
        max_length=args.max_length,
        opset_version=args.opset_version
    )


if __name__ == "__main__":
    main()
