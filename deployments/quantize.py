"""
模型量化脚本
支持 PTQ (Post-Training Quantization) 到 INT8
针对 3050Ti 优化
"""

import os
import sys
import argparse
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, str(Path(__file__).parent.parent))


def quantize_onnx_model(
    input_path: str,
    output_path: str,
    optimize_model: bool = True
):
    """
    对 ONNX 模型进行 INT8 动态量化
    
    Args:
        input_path: 输入 ONNX 模型路径
        output_path: 输出量化模型路径
        optimize_model: 是否先进行图优化
    """
    print(f"正在加载 ONNX 模型：{input_path}")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 先进行图优化（可选）
    if optimize_model:
        print("正在进行图优化...")
        try:
            import onnxoptimizer
            model = onnx.load(str(input_path))
            optimized_model = onnxoptimizer.optimize(model)
            
            optimized_path = input_path.parent / 'model_optimized.onnx'
            onnx.save(optimized_model, str(optimized_path))
            print(f"优化后的模型已保存到：{optimized_path}")
            input_path = optimized_path
        except ImportError:
            print("未安装 onnxoptimizer，跳过图优化")
    
    # 动态量化
    print(f"正在进行 INT8 量化...")
    
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8  # 使用 INT8
        # optimize_model 参数已在新版本中移除，图优化已在上面完成
    )
    
    print(f"量化完成！")
    
    # 比较模型大小
    original_size = Path(input_path).stat().st_size / (1024 * 1024)
    quantized_size = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n模型大小对比：")
    print(f"  原始模型：{original_size:.2f} MB")
    print(f"  量化模型：{quantized_size:.2f} MB")
    print(f"  压缩比例：{(1 - quantized_size/original_size) * 100:.1f}%")
    
    # 验证量化模型
    try:
        model = onnx.load(str(output_path))
        onnx.checker.check_model(model)
        print(f"\n量化模型验证通过！")
    except Exception as e:
        print(f"\n量化模型验证失败：{e}")


def benchmark_models(
    fp32_path: str,
    int8_path: str,
    num_runs: int = 100
):
    """
    对比 FP32 和 INT8 模型的推理速度
    
    Args:
        fp32_path: FP32 模型路径
        int8_path: INT8 模型路径
        num_runs: 测试次数
    """
    print("\n正在对比推理性能...")
    
    try:
        import onnxruntime as ort
        import numpy as np
        import time
        
        # 创建测试输入
        batch_size = 1
        seq_length = 128
        
        input_ids = np.random.randint(0, 10000, (batch_size, seq_length), dtype=np.int64)
        attention_mask = np.ones((batch_size, seq_length), dtype=np.int64)
        token_type_ids = np.zeros((batch_size, seq_length), dtype=np.int64)
        
        # 测试 FP32 模型
        print("\n测试 FP32 模型...")
        sess_fp32 = ort.InferenceSession(str(fp32_path))
        
        # 预热
        for _ in range(10):
            sess_fp32.run(None, {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            })
        
        # 正式测试
        start_time = time.time()
        for _ in range(num_runs):
            sess_fp32.run(None, {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            })
        fp32_time = (time.time() - start_time) / num_runs * 1000  # ms
        
        # 测试 INT8 模型
        print("测试 INT8 模型...")
        sess_int8 = ort.InferenceSession(str(int8_path))
        
        # 预热
        for _ in range(10):
            sess_int8.run(None, {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            })
        
        # 正式测试
        start_time = time.time()
        for _ in range(num_runs):
            sess_int8.run(None, {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            })
        int8_time = (time.time() - start_time) / num_runs * 1000  # ms
        
        print(f"\n性能对比（平均延迟）：")
        print(f"  FP32 模型：{fp32_time:.2f} ms")
        print(f"  INT8 模型：{int8_time:.2f} ms")
        print(f"  加速比：{fp32_time/int8_time:.2f}x")
        
    except ImportError:
        print("未安装 onnxruntime，跳过性能测试")
    except Exception as e:
        print(f"性能测试失败：{e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='模型量化')
    parser.add_argument('--input_path', type=str, default='deployments/model.onnx',
                        help='输入 ONNX 模型路径')
    parser.add_argument('--output_path', type=str, default='deployments/model_int8.onnx',
                        help='输出量化模型路径')
    parser.add_argument('--skip_optimization', action='store_true',
                        help='跳过图优化')
    parser.add_argument('--benchmark', action='store_true',
                        help='是否进行性能对比')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='性能测试次数')
    
    args = parser.parse_args()
    
    # 量化模型
    quantize_onnx_model(
        input_path=args.input_path,
        output_path=args.output_path,
        optimize_model=not args.skip_optimization
    )
    
    # 性能对比
    if args.benchmark:
        benchmark_models(
            fp32_path=args.input_path,
            int8_path=args.output_path,
            num_runs=args.num_runs
        )


if __name__ == "__main__":
    main()
