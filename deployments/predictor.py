"""
基于 TensorRT/ONNX Runtime 的高速推理类
针对 3050Ti 优化，目标延迟 < 0.5ms
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PredictionResult:
    """预测结果"""
    text: str
    aspect_scores: Dict[str, Dict[str, float]]  # 每个维度的分数
    overall_sentiment: str  # 整体情感
    confidence: float  # 置信度
    inference_time_ms: float  # 推理时间


class SentimentPredictor:
    """
    情感分析预测器
    支持 ONNX Runtime 和 TensorRT Provider
    """
    
    # 维度名称
    ASPECT_NAMES = [
        '整体', '性价比', '质量', '外观', '功能', '服务',
        '物流', '包装', '售后', '态度', '速度', '专业度',
        '体验', '舒适度', '耐用性', '易用性', '效果', '其他'
    ]
    
    # 情感标签
    SENTIMENT_LABELS = {0: '负面', 1: '中性', 2: '正面'}
    
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        use_tensorrt: bool = True,
        max_length: int = 128,
        device_id: int = 0
    ):
        """
        Args:
            model_path: ONNX 模型路径
            tokenizer_path: 分词器路径（默认与模型同目录）
            use_tensorrt: 是否使用 TensorRT Provider
            max_length: 最大序列长度
            device_id: GPU 设备 ID
        """
        self.model_path = Path(model_path)
        self.max_length = max_length
        self.use_tensorrt = use_tensorrt
        
        # 加载分词器
        if tokenizer_path is None:
            tokenizer_path = self.model_path.parent.parent / 'checkpoints' / 'best_model'
        
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 创建 ONNX Runtime Session
        self._create_session(device_id)
        
        # 获取输入输出名称
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"预测器初始化完成")
        print(f"  模型：{model_path}")
        print(f"  Provider：{self.session.get_providers()}")
    
    def _create_session(self, device_id: int):
        """创建 ONNX Runtime Session"""
        import onnxruntime as ort
        
        # 配置 Provider
        providers = []
        
        if self.use_tensorrt and ort.get_device() == 'GPU':
            # TensorRT Provider 配置（针对 3050Ti 优化）
            trt_config = {
                'device_id': device_id,
                'trt_max_workspace_size': 2147483648,  # 2GB
                'trt_fp16_enable': True,  # 启用 FP16
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(self.model_path.parent / 'trt_cache')
            }
            providers.append(('TensorrtExecutionProvider', trt_config))
            providers.append('CUDAExecutionProvider')
        
        if ort.get_device() == 'GPU':
            providers.append('CUDAExecutionProvider')
        
        providers.append('CPUExecutionProvider')
        
        # Session 配置
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # 优化 CPU 线程数
        
        # 创建 Session
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )
    
    def predict(self, text: str) -> PredictionResult:
        """
        单条预测
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果
        """
        # 编码
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='np'
        )
        
        # 准备输入
        ort_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64),
        }
        
        # 添加 token_type_ids（如果模型需要）
        if 'token_type_ids' in self.input_names:
            if 'token_type_ids' in inputs:
                ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)
            else:
                ort_inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'], dtype=np.int64)
        
        # 推理
        start_time = time.time()
        outputs = self.session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # 解析输出
        logits = outputs[0]  # [1, num_labels, num_classes]
        probs = self._softmax(logits[0])  # [num_labels, num_classes]
        
        # 构建结果
        aspect_scores = {}
        for i, aspect_name in enumerate(self.ASPECT_NAMES):
            if i < len(probs):
                aspect_scores[aspect_name] = {
                    self.SENTIMENT_LABELS[j]: float(probs[i][j])
                    for j in range(len(self.SENTIMENT_LABELS))
                }
        
        # 整体情感（基于第一个维度"整体"）
        overall_idx = np.argmax(probs[0])
        overall_sentiment = self.SENTIMENT_LABELS[overall_idx]
        confidence = float(probs[0][overall_idx])
        
        return PredictionResult(
            text=text,
            aspect_scores=aspect_scores,
            overall_sentiment=overall_sentiment,
            confidence=confidence,
            inference_time_ms=inference_time
        )
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[PredictionResult]:
        """
        批量预测
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 编码
            inputs = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors='np'
            )
            
            # 准备输入
            ort_inputs = {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64),
            }
            
            if 'token_type_ids' in self.input_names:
                if 'token_type_ids' in inputs:
                    ort_inputs['token_type_ids'] = inputs['token_type_ids'].astype(np.int64)
                else:
                    ort_inputs['token_type_ids'] = np.zeros_like(inputs['input_ids'], dtype=np.int64)
            
            # 推理
            start_time = time.time()
            outputs = self.session.run(None, ort_inputs)
            inference_time = (time.time() - start_time) * 1000 / len(batch_texts)
            
            # 解析结果
            logits = outputs[0]  # [batch_size, num_labels, num_classes]
            probs = self._softmax(logits)
            
            for j, text in enumerate(batch_texts):
                aspect_scores = {}
                for k, aspect_name in enumerate(self.ASPECT_NAMES):
                    if k < len(probs[j]):
                        aspect_scores[aspect_name] = {
                            self.SENTIMENT_LABELS[m]: float(probs[j][k][m])
                            for m in range(len(self.SENTIMENT_LABELS))
                        }
                
                overall_idx = np.argmax(probs[j][0])
                overall_sentiment = self.SENTIMENT_LABELS[overall_idx]
                confidence = float(probs[j][0][overall_idx])
                
                results.append(PredictionResult(
                    text=text,
                    aspect_scores=aspect_scores,
                    overall_sentiment=overall_sentiment,
                    confidence=confidence,
                    inference_time_ms=inference_time
                ))
        
        return results
    
    def benchmark(self, num_runs: int = 1000) -> Dict:
        """
        性能基准测试
        
        Args:
            num_runs: 测试次数
            
        Returns:
            性能指标
        """
        test_text = "这个产品质量很好，物流也很快，非常满意！"
        
        # 预热
        for _ in range(100):
            self.predict(test_text)
        
        # 测试
        times = []
        for _ in range(num_runs):
            result = self.predict(test_text)
            times.append(result.inference_time_ms)
        
        times = np.array(times)
        
        metrics = {
            'mean_latency_ms': float(np.mean(times)),
            'p50_latency_ms': float(np.percentile(times, 50)),
            'p95_latency_ms': float(np.percentile(times, 95)),
            'p99_latency_ms': float(np.percentile(times, 99)),
            'min_latency_ms': float(np.min(times)),
            'max_latency_ms': float(np.max(times)),
            'throughput_qps': 1000.0 / float(np.mean(times))
        }
        
        print("\n性能基准测试结果：")
        print(f"  平均延迟：{metrics['mean_latency_ms']:.3f} ms")
        print(f"  P50 延迟：{metrics['p50_latency_ms']:.3f} ms")
        print(f"  P95 延迟：{metrics['p95_latency_ms']:.3f} ms")
        print(f"  P99 延迟：{metrics['p99_latency_ms']:.3f} ms")
        print(f"  最小延迟：{metrics['min_latency_ms']:.3f} ms")
        print(f"  最大延迟：{metrics['max_latency_ms']:.3f} ms")
        print(f"  吞吐量：{metrics['throughput_qps']:.1f} QPS")
        
        return metrics
    
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Softmax 计算"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='情感分析推理')
    parser.add_argument('--model_path', type=str, default='deployments/model_int8.onnx',
                        help='ONNX 模型路径')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='分词器路径')
    parser.add_argument('--use_tensorrt', action='store_true',
                        help='使用 TensorRT')
    parser.add_argument('--text', type=str, default=None,
                        help='预测文本')
    parser.add_argument('--benchmark', action='store_true',
                        help='性能测试')
    parser.add_argument('--num_runs', type=int, default=1000,
                        help='性能测试次数')
    
    args = parser.parse_args()
    
    # 创建预测器
    predictor = SentimentPredictor(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        use_tensorrt=args.use_tensorrt
    )
    
    # 单条预测
    if args.text:
        result = predictor.predict(args.text)
        print(f"\n输入文本：{result.text}")
        print(f"整体情感：{result.overall_sentiment} (置信度: {result.confidence:.3f})")
        print(f"推理时间：{result.inference_time_ms:.3f} ms")
        print(f"\n各维度详情：")
        for aspect, scores in list(result.aspect_scores.items())[:5]:
            pred_label = max(scores, key=scores.get)
            print(f"  {aspect}: {pred_label} (置信度: {scores[pred_label]:.3f})")
    
    # 性能测试
    if args.benchmark:
        predictor.benchmark(num_runs=args.num_runs)


if __name__ == "__main__":
    main()
