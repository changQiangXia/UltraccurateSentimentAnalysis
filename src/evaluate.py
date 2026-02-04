"""
评估脚本
多维度评估指标：Accuracy, F1-score, MAE, 混淆矩阵
"""

import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, mean_absolute_error
)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SentimentAnalyzer
from transformers import AutoTokenizer
from src.train import SentimentDataset
from torch.utils.data import DataLoader


class Evaluator:
    """评估器"""
    
    # ASAP 18 维度名称
    ASPECT_NAMES = [
        '整体', '性价比', '质量', '外观', '功能', '服务',
        '物流', '包装', '售后', '态度', '速度', '专业度',
        '体验', '舒适度', '耐用性', '易用性', '效果', '其他'
    ]
    
    def __init__(self, model_path: str, config: Dict = None):
        """
        Args:
            model_path: 模型路径
            config: 配置字典
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # 加载配置
        if config is None:
            with open(self.model_path / 'config.json', 'r', encoding='utf-8') as f:
                model_config = json.load(f)
        else:
            model_config = {
                'model_name': config['model']['name'],
                'num_labels': config['model']['num_labels'],
                'hidden_dropout_prob': config['model']['hidden_dropout_prob']
            }
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载模型
        self.model = SentimentAnalyzer(
            model_name=model_config['model_name'],
            num_labels=model_config['num_labels'],
            dropout=model_config.get('hidden_dropout_prob', 0.1)
        )
        
        # 加载权重
        state_dict = torch.load(self.model_path / 'pytorch_model.bin', map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"模型加载完成：{model_path}")
    
    def evaluate(self, data_path: str, batch_size: int = 32, save_report: bool = True) -> Dict:
        """
        评估模型
        
        Args:
            data_path: 数据文件路径
            batch_size: 批次大小
            save_report: 是否保存详细报告
            
        Returns:
            评估指标字典
        """
        # 创建数据集
        dataset = SentimentDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=128
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 收集预测结果
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                logits = outputs['logits']
                probs = outputs['probabilities']
                
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        
        # 合并结果
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # 计算指标
        metrics = self._compute_detailed_metrics(all_preds, all_labels, all_probs)
        
        # 保存报告
        if save_report:
            self._save_report(metrics, data_path)
        
        return metrics
    
    def _compute_detailed_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray
    ) -> Dict:
        """
        计算详细评估指标
        
        Args:
            preds: 预测标签 [num_samples, num_labels]
            labels: 真实标签 [num_samples, num_labels]
            probs: 预测概率 [num_samples, num_labels, num_classes]
            
        Returns:
            详细指标字典
        """
        num_labels = preds.shape[1]
        
        # 每个维度的指标
        dim_metrics = []
        for i in range(num_labels):
            metrics_i = {
                'aspect': self.ASPECT_NAMES[i] if i < len(self.ASPECT_NAMES) else f'Aspect_{i}',
                'accuracy': accuracy_score(labels[:, i], preds[:, i]),
                'precision_macro': precision_score(labels[:, i], preds[:, i], average='macro', zero_division=0),
                'recall_macro': recall_score(labels[:, i], preds[:, i], average='macro', zero_division=0),
                'f1_macro': f1_score(labels[:, i], preds[:, i], average='macro', zero_division=0),
                'f1_weighted': f1_score(labels[:, i], preds[:, i], average='weighted', zero_division=0),
            }
            
            # MAE（将标签映射为 -1, 0, 1 后计算）
            labels_mapped = labels[:, i].astype(float) - 1  # 0->-1, 1->0, 2->1
            preds_mapped = preds[:, i].astype(float) - 1
            metrics_i['mae'] = mean_absolute_error(labels_mapped, preds_mapped)
            
            dim_metrics.append(metrics_i)
        
        # 整体指标
        overall_metrics = {
            'strict_accuracy': np.all(preds == labels, axis=1).mean(),  # 所有维度都正确
            'avg_dim_accuracy': np.mean([m['accuracy'] for m in dim_metrics]),
            'avg_f1_macro': np.mean([m['f1_macro'] for m in dim_metrics]),
            'avg_f1_weighted': np.mean([m['f1_weighted'] for m in dim_metrics]),
            'avg_mae': np.mean([m['mae'] for m in dim_metrics]),
        }
        
        # 计算混淆矩阵
        confusion_matrices = []
        for i in range(num_labels):
            cm = confusion_matrix(labels[:, i], preds[:, i], labels=[0, 1, 2])
            confusion_matrices.append(cm)
        
        return {
            'overall': overall_metrics,
            'per_dimension': dim_metrics,
            'confusion_matrices': confusion_matrices,
            'predictions': preds,
            'labels': labels,
            'probabilities': probs
        }
    
    def _save_report(self, metrics: Dict, data_path: str):
        """保存评估报告"""
        report_dir = Path('reports')
        report_dir.mkdir(exist_ok=True)
        
        # 保存 JSON 报告
        report_file = report_dir / 'evaluation_report.json'
        
        # 转换为可序列化的格式
        report_data = {
            'overall': metrics['overall'],
            'per_dimension': metrics['per_dimension']
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 保存 CSV 详细报告
        df = pd.DataFrame(metrics['per_dimension'])
        csv_file = report_dir / 'per_dimension_metrics.csv'
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 打印报告
        print("\n" + "="*60)
        print("评估报告")
        print("="*60)
        print(f"\n整体指标：")
        for key, value in metrics['overall'].items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\n各维度详细指标（前 5 个）：")
        for i, dim in enumerate(metrics['per_dimension'][:5]):
            print(f"\n  {dim['aspect']}:")
            print(f"    Accuracy: {dim['accuracy']:.4f}")
            print(f"    F1 (macro): {dim['f1_macro']:.4f}")
            print(f"    MAE: {dim['mae']:.4f}")
        
        print(f"\n报告已保存到：{report_dir}")
    
    def plot_confusion_matrix(self, aspect_idx: int = 0, save_path: str = None):
        """
        绘制混淆矩阵
        
        Args:
            aspect_idx: 维度索引
            save_path: 保存路径
        """
        pass  # 实际使用时实现可视化


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估情感分析模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径')
    parser.add_argument('--test_file', type=str, required=True,
                        help='测试文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    
    args = parser.parse_args()
    
    # 加载配置（可选）
    config = None
    config_path = Path(args.model_path) / '..' / '..' / 'configs' / 'hyperparams.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    
    # 创建评估器
    evaluator = Evaluator(args.model_path, config)
    
    # 评估
    metrics = evaluator.evaluate(args.test_file, args.batch_size)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()
