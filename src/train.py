"""
训练脚本
支持 R-Drop、混合精度训练 (FP16)、断点续训
针对 4GB 显存优化
"""

import os
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import SentimentAnalyzer, load_model_and_tokenizer
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """情感分析数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        """
        Args:
            data_path: jsonl 文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        labels = item['labels']
        
        # 编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class Trainer:
    """训练器（支持断点续训）"""
    
    def __init__(self, config: Dict, resume: bool = False):
        """
        Args:
            config: 配置字典
            resume: 是否从 checkpoint 恢复
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('-inf')
        
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型和分词器
        logger.info(f"加载模型: {config['model']['name']}")
        
        # 获取类别加权配置
        use_weighted_loss = config['model'].get('use_weighted_loss', True)
        class_weights = config['model'].get('class_weights', None)
        
        if use_weighted_loss and class_weights:
            logger.info(f"使用类别加权 Loss: {class_weights}")
        
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=config['model']['name'],
            num_labels=config['model']['num_labels'],
            dropout=config['model']['hidden_dropout_prob'],
            use_rdrop=config['rdrop']['enabled'],
            rdrop_alpha=config['rdrop']['alpha'],
            use_weighted_loss=use_weighted_loss,
            class_weights=class_weights
        )
        self.model.to(self.device)
        
        # 创建输出目录
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 混合精度训练
        self.use_fp16 = config['training'].get('fp16', False) and torch.cuda.is_available()
        if self.use_fp16:
            try:
                from torch.amp import GradScaler
                self.scaler = GradScaler('cuda')
            except ImportError:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            logger.info("启用混合精度训练 (FP16)")
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器（将在 train 中初始化）
        self.scheduler = None
        
        # 断点续训
        if resume:
            self.resume_from_checkpoint()
    
    def _create_optimizer(self):
        """创建优化器（支持层 wise 学习率衰减）"""
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 层 wise 学习率衰减配置
        layer_decay = self.config['training'].get('layer_wise_lr_decay', 0.95)
        base_lr = self.config['training']['learning_rate']
        
        # 分组参数
        optimizer_grouped_parameters = []
        
        # BERT 层（从底层到顶层递减学习率）
        num_layers = self.model.bert.config.num_hidden_layers
        
        for layer_num in range(num_layers):
            # 层号越大（越靠近输出），学习率越高
            layer_lr = base_lr * (layer_decay ** (num_layers - layer_num - 1))
            
            # 该层的参数
            layer_params_decay = []
            layer_params_no_decay = []
            
            for n, p in self.model.named_parameters():
                if f'encoder.layer.{layer_num}.' in n:
                    if any(nd in n for nd in no_decay):
                        layer_params_no_decay.append(p)
                    else:
                        layer_params_decay.append(p)
            
            if layer_params_decay:
                optimizer_grouped_parameters.append({
                    'params': layer_params_decay,
                    'lr': layer_lr,
                    'weight_decay': self.config['training']['weight_decay']
                })
            if layer_params_no_decay:
                optimizer_grouped_parameters.append({
                    'params': layer_params_no_decay,
                    'lr': layer_lr,
                    'weight_decay': 0.0
                })
        
        # 输出层（分类头）使用最大学习率
        classifier_params_decay = []
        classifier_params_no_decay = []
        
        for n, p in self.model.named_parameters():
            if 'classifiers' in n or 'classifier' in n:
                if any(nd in n for nd in no_decay):
                    classifier_params_no_decay.append(p)
                else:
                    classifier_params_decay.append(p)
        
        if classifier_params_decay:
            optimizer_grouped_parameters.append({
                'params': classifier_params_decay,
                'lr': base_lr,  # 最高学习率
                'weight_decay': self.config['training']['weight_decay']
            })
        if classifier_params_no_decay:
            optimizer_grouped_parameters.append({
                'params': classifier_params_no_decay,
                'lr': base_lr,
                'weight_decay': 0.0
            })
        
        # 使用 AdamW
        from torch.optim import AdamW
        optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr)
        
        # 打印各层学习率
        print("\n层 wise 学习率配置:")
        for i, group in enumerate(optimizer_grouped_parameters[:6]):  # 只打印前6组
            print(f"  Group {i}: lr={group['lr']:.2e}, decay={group['weight_decay']}")
        
        return optimizer
    
    def _create_dataloader(self, data_path: str, shuffle: bool = True) -> DataLoader:
        """创建 DataLoader"""
        dataset = SentimentDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config['model']['max_length']
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['per_device_train_batch_size'],
            shuffle=shuffle,
            num_workers=self.config['training'].get('dataloader_num_workers', 0),
            pin_memory=self.config['training'].get('dataloader_pin_memory', False)
        )
    
    def resume_from_checkpoint(self):
        """从 checkpoint 恢复训练状态"""
        checkpoint_dir = self.output_dir / 'checkpoint-latest'
        
        if not checkpoint_dir.exists():
            logger.warning(f"没有找到 checkpoint: {checkpoint_dir}")
            logger.warning("将从头开始训练")
            return
        
        logger.info(f"从 checkpoint 恢复: {checkpoint_dir}")
        
        # 加载模型权重
        model_path = checkpoint_dir / 'pytorch_model.bin'
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("模型权重加载成功")
        
        # 加载训练状态
        trainer_state_path = checkpoint_dir / 'trainer_state.json'
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.global_step = state.get('global_step', 0)
            self.current_epoch = state.get('current_epoch', 0)
            self.best_metric = state.get('best_metric', float('-inf'))
            
            logger.info(f"恢复训练状态: epoch={self.current_epoch}, step={self.global_step}, best_metric={self.best_metric:.4f}")
        
        # 加载优化器状态
        optimizer_path = checkpoint_dir / 'optimizer.pt'
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state)
            logger.info("优化器状态加载成功")
        
        # 加载 scheduler 状态（如果存在）
        scheduler_path = checkpoint_dir / 'scheduler.pt'
        if scheduler_path.exists():
            self.scheduler_state = torch.load(scheduler_path)
            logger.info("学习率调度器状态加载成功")
        else:
            self.scheduler_state = None
        
        logger.info("断点续训准备完成！")
    
    def train(self, train_path: str, eval_path: str):
        """
        训练模型
        
        Args:
            train_path: 训练集路径
            eval_path: 验证集路径
        """
        train_loader = self._create_dataloader(train_path, shuffle=True)
        eval_loader = self._create_dataloader(eval_path, shuffle=False)
        
        # 计算总训练步数
        epochs = self.config['training']['num_train_epochs']
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(total_steps * self.config['training']['warmup_ratio'])
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 如果从 checkpoint 恢复，加载 scheduler 状态
        if hasattr(self, 'scheduler_state') and self.scheduler_state is not None:
            self.scheduler.load_state_dict(self.scheduler_state)
            logger.info("学习率调度器状态已恢复")
        
        logger.info(f"总训练步数: {total_steps}, Warmup 步数: {warmup_steps}")
        logger.info(f"每 epoch 步数: {steps_per_epoch}")
        
        # 如果从 checkpoint 恢复，跳过已训练的 epoch
        start_epoch = self.current_epoch
        if start_epoch > 0:
            logger.info(f"从 epoch {start_epoch + 1} 继续训练（已跳过前 {start_epoch} 个 epoch）")
        
        # Early Stopping 设置
        patience = self.config['training'].get('early_stopping_patience', 5)
        no_improve_count = 0
        metric_for_best = self.config['training']['metric_for_best_model']
        
        # 训练循环
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            logger.info(f"\n===== Epoch {epoch + 1}/{epochs} =====")
            
            # 训练阶段
            train_loss = self._train_epoch(train_loader, steps_per_epoch)
            logger.info(f"训练损失: {train_loss:.4f}")
            
            # 验证阶段
            eval_metrics = self._eval_epoch(eval_loader)
            logger.info(f"验证指标: {eval_metrics}")
            
            # 保存最佳模型
            current_metric = eval_metrics.get(metric_for_best, 0)
            
            if current_metric > self.best_metric:
                improvement = current_metric - self.best_metric
                self.best_metric = current_metric
                self._save_model('best_model')
                logger.info(f"🎉 新最佳模型！{metric_for_best}: {current_metric:.4f} (+{improvement:.4f})")
                no_improve_count = 0  # 重置计数器
            else:
                no_improve_count += 1
                logger.info(f"未提升 ({no_improve_count}/{patience})")
            
            # 保存 checkpoint（用于断点续训）
            self._save_checkpoint('checkpoint-latest')
            
            # 定期保存历史 checkpoint
            if (epoch + 1) % 1 == 0:
                self._save_checkpoint(f'checkpoint-epoch-{epoch + 1}')
            
            # Early Stopping 检查
            if no_improve_count >= patience:
                logger.info(f"\n⏹️ Early Stopping: {patience} 个 epoch 无提升，停止训练")
                break
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info("\n训练完成！")
        logger.info(f"最佳 {metric_for_best}: {self.best_metric:.4f}")
        logger.info(f"总训练步数: {self.global_step}")
    
    def _train_epoch(self, dataloader: DataLoader, steps_per_epoch: int) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in progress_bar:
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            if self.use_fp16:
                try:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        outputs = self.model.rdrop_forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels
                        )
                        loss = outputs['loss']
                except ImportError:
                    from torch.cuda.amp import autocast
                    with autocast():
                        outputs = self.model.rdrop_forward(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels
                        )
                        loss = outputs['loss']
                
                # 反向传播（混合精度）
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model.rdrop_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss = outputs['loss']
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                'step': self.global_step
            })
            
            # 定期记录
            if self.global_step % self.config['training']['logging_steps'] == 0:
                logger.info(f"Step {self.global_step}: loss={loss.item():.4f}")
        
        return total_loss / len(dataloader)
    
    def _eval_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证一个 epoch"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # 预测
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                labels_np = labels.cpu().numpy()
                
                all_preds.append(preds)
                all_labels.append(labels_np)
        
        # 合并所有预测
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # 计算指标
        metrics = self._compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            preds: 预测结果 [num_samples, num_labels]
            labels: 真实标签 [num_samples, num_labels]
            
        Returns:
            指标字典
        """
        # 每个维度的准确率
        dim_accuracies = []
        for i in range(preds.shape[1]):
            acc = accuracy_score(labels[:, i], preds[:, i])
            dim_accuracies.append(acc)
        
        # 整体准确率（所有维度都正确）
        overall_acc = np.all(preds == labels, axis=1).mean()
        
        # 平均 F1（macro）
        f1_macros = []
        for i in range(preds.shape[1]):
            f1 = f1_score(labels[:, i], preds[:, i], average='macro', zero_division=0)
            f1_macros.append(f1)
        
        metrics = {
            'accuracy': overall_acc,
            'avg_dim_accuracy': np.mean(dim_accuracies),
            'f1_macro': np.mean(f1_macros),
            'avg_dim_f1': np.mean(f1_macros)
        }
        
        return metrics
    
    def _save_checkpoint(self, name: str):
        """保存完整 checkpoint（用于断点续训）"""
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), save_dir / 'pytorch_model.bin')
        
        # 保存训练状态
        trainer_state = {
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'config': self.config
        }
        with open(save_dir / 'trainer_state.json', 'w', encoding='utf-8') as f:
            json.dump(trainer_state, f, ensure_ascii=False, indent=2)
        
        # 保存优化器状态
        torch.save(self.optimizer.state_dict(), save_dir / 'optimizer.pt')
        
        # 保存 scheduler 状态
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), save_dir / 'scheduler.pt')
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_dir)
        
        logger.info(f"Checkpoint 已保存: {save_dir} (epoch={self.current_epoch + 1}, step={self.global_step})")
    
    def _save_model(self, name: str):
        """保存模型（简洁版本，用于部署）"""
        save_dir = self.output_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), save_dir / 'pytorch_model.bin')
        
        # 保存配置
        config = {
            'model_name': self.config['model']['name'],
            'num_labels': self.config['model']['num_labels'],
            'hidden_dropout_prob': self.config['model']['hidden_dropout_prob']
        }
        with open(save_dir / 'config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 保存分词器
        self.tokenizer.save_pretrained(save_dir)
        
        logger.info(f"模型已保存: {save_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练情感分析模型')
    parser.add_argument('--config', type=str, default='configs/hyperparams.yaml',
                        help='配置文件路径')
    parser.add_argument('--train_file', type=str, default=None,
                        help='训练文件路径')
    parser.add_argument('--eval_file', type=str, default=None,
                        help='验证文件路径')
    parser.add_argument('--resume', action='store_true',
                        help='从 checkpoint 断点续训')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 使用命令行参数覆盖配置
    if args.train_file:
        config['data']['train_file'] = args.train_file
    if args.eval_file:
        config['data']['eval_file'] = args.eval_file
    
    # 创建训练器
    trainer = Trainer(config, resume=args.resume)
    
    # 开始训练
    trainer.train(
        train_path=config['data']['train_file'],
        eval_path=config['data']['eval_file']
    )


if __name__ == "__main__":
    main()
