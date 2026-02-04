"""
模型架构定义
包含 Chinese-TinyBERT-L4 封装和 R-Drop Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, Optional, Tuple


class SentimentAnalyzer(nn.Module):
    """
    基于 Chinese-TinyBERT-L4 的情感分析模型
    支持多标签分类（ASAP 18 维度）
    """
    
    def __init__(
        self,
        model_name: str = "hfl/chinese-tinybert-l4-uncased",
        num_labels: int = 18,
        num_aspect_classes: int = 3,  # 每个维度：负/中/正
        dropout: float = 0.1,
        use_rdrop: bool = True,
        rdrop_alpha: float = 5.0,
        use_weighted_loss: bool = True,  # 是否使用加权 Loss
        class_weights: list = None  # 类别权重 [负, 中, 正]
    ):
        """
        Args:
            model_name: 预训练模型名称
            num_labels: 标签维度数（ASAP 为 18）
            num_aspect_classes: 每个维度的类别数
            dropout: dropout 概率
            use_rdrop: 是否使用 R-Drop
            rdrop_alpha: R-Drop 损失权重
            use_weighted_loss: 是否使用类别加权 Loss（解决数据不平衡）
            class_weights: 类别权重，默认 [5.0, 1.0, 2.5] 针对 ASAP 数据分布优化
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.num_aspect_classes = num_aspect_classes
        self.use_rdrop = use_rdrop
        self.rdrop_alpha = rdrop_alpha
        self.use_weighted_loss = use_weighted_loss
        
        # 设置类别权重（针对 ASAP 数据分布：负面 4%，中性 76%，正面 20%）
        if class_weights is None:
            # 根据频率倒数设置权重：中性 1.0，正面 3.8，负面 19.0 → 归一化后约 [5.0, 1.0, 2.5]
            self.class_weights = torch.tensor([5.0, 1.0, 2.5])
        else:
            self.class_weights = torch.tensor(class_weights)
        
        # 判断是否为本地路径
        from pathlib import Path
        is_local = Path(model_name).exists()
        
        # 加载预训练配置
        if is_local:
            self.config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        else:
            self.config = AutoConfig.from_pretrained(model_name)
        
        self.config.hidden_dropout_prob = dropout
        self.config.attention_probs_dropout_prob = dropout
        
        # 加载预训练模型
        if is_local:
            self.bert = AutoModel.from_pretrained(model_name, config=self.config, local_files_only=True)
        else:
            self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # 可选：开启 Gradient Checkpointing（节省 30-40% 显存，但训练慢 20%）
        # self.bert.gradient_checkpointing_enable()
        
        hidden_size = self.config.hidden_size
        
        # 多任务分类头：每个维度一个分类器
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_size, num_aspect_classes)
            for _ in range(num_labels)
        ])
        
        # 初始化分类头
        for classifier in self.classifiers:
            nn.init.normal_(classifier.weight, std=0.02)
            nn.init.zeros_(classifier.bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 将权重注册为 buffer（不计算梯度，但会随模型保存）
        if self.use_weighted_loss:
            self.register_buffer('class_weights_buffer', self.class_weights)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token ids
            attention_mask: 注意力掩码
            token_type_ids: token 类型 ids
            labels: 标签 [batch_size, num_labels]，每个维度 0/1/2
            
        Returns:
            包含 logits、loss、probabilities 的字典
        """
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 取 [CLS] token 的表示
        pooled_output = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # 每个维度的 logits
        logits_list = []
        for classifier in self.classifiers:
            logits_list.append(classifier(pooled_output))  # [batch_size, num_aspect_classes]
        
        # 堆叠为 [batch_size, num_labels, num_aspect_classes]
        logits = torch.stack(logits_list, dim=1)
        
        # 计算概率
        probs = F.softmax(logits, dim=-1)
        
        result = {
            'logits': logits,
            'probabilities': probs,
            'hidden_states': outputs.last_hidden_state
        }
        
        # 计算损失
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            result['loss'] = loss
        
        return result
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算多标签分类损失
        
        Args:
            logits: [batch_size, num_labels, num_aspect_classes]
            labels: [batch_size, num_labels]
            
        Returns:
            损失值
        """
        batch_size, num_labels, num_classes = logits.shape
        
        # 重塑为交叉熵需要的形状
        logits_flat = logits.view(-1, num_classes)  # [batch_size * num_labels, num_classes]
        labels_flat = labels.view(-1)  # [batch_size * num_labels]
        
        # 使用 Focal Loss 或加权交叉熵损失
        if self.use_weighted_loss and hasattr(self, 'class_weights_buffer'):
            # Focal Loss（对难分类样本更关注）
            from src.losses import FocalLoss
            alpha = self.class_weights_buffer.cpu().numpy().tolist()
            focal_loss_fn = FocalLoss(num_classes=num_classes, alpha=alpha, gamma=2.0)
            loss = focal_loss_fn(logits_flat, labels_flat)
        else:
            # 普通交叉熵
            loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
        
        return loss
    
    def rdrop_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        R-Drop 前向传播：对同一个样本进行两次 forward，约束概率分布一致
        
        Args:
            input_ids: 输入 token ids
            attention_mask: 注意力掩码
            token_type_ids: token 类型 ids
            labels: 标签
            
        Returns:
            包含 loss、logits 的字典
        """
        # 第一次 forward
        outputs1 = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None
        )
        
        # 第二次 forward（dropout 会随机不同）
        outputs2 = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None
        )
        
        logits1 = outputs1['logits']
        logits2 = outputs2['logits']
        probs1 = outputs1['probabilities']
        probs2 = outputs2['probabilities']
        
        # 计算 R-Drop 损失
        loss = 0.0
        
        # 1. 任务损失（使用第一次的 logits）
        if labels is not None:
            task_loss = self.compute_loss(logits1, labels)
            loss = loss + task_loss
        
        # 2. R-Drop 一致性损失（KL 散度）
        if self.use_rdrop and self.training:
            # 对每个维度计算 KL 散度
            kl_loss = 0.0
            for i in range(self.num_labels):
                p1 = probs1[:, i, :]  # [batch_size, num_classes]
                p2 = probs2[:, i, :]
                # 对称 KL 散度
                kl = F.kl_div((p1 + 1e-8).log(), p2, reduction='batchmean')
                kl = kl + F.kl_div((p2 + 1e-8).log(), p1, reduction='batchmean')
                kl_loss = kl_loss + kl / 2
            
            kl_loss = kl_loss / self.num_labels
            loss = loss + self.rdrop_alpha * kl_loss
        
        return {
            'loss': loss,
            'logits': logits1,
            'probabilities': probs1
        }


class LLMJudge:
    """
    LLM-as-a-Judge：用大模型过滤噪声样本
    注意：实际使用时需要配置 API 或本地模型
    """
    
    def __init__(self, model_name: str = "qwen2-7b-instruct"):
        self.model_name = model_name
        self.prompt_template = """请判断以下评论与其情感标签是否匹配。

评论：{text}

标签维度（共18维，每维0=负面，1=中性，2=正面）：
{labels}

请分析：
1. 标签是否与评论内容一致？
2. 如果不一致，可能的正确标签是什么？
3. 给出置信度（0-1）：

以JSON格式返回：
{{"consistent": true/false, "confidence": 0.8, "reason": "..."}}"""

    def judge_sample(self, text: str, labels: list) -> dict:
        """
        判断单条样本质量
        
        Args:
            text: 评论文本
            labels: 18维标签列表
            
        Returns:
            判断结果
        """
        # 这里应该调用实际的大模型 API
        # 示例返回
        return {
            "consistent": True,
            "confidence": 0.9,
            "reason": "标签与文本内容基本匹配"
        }
    
    def filter_dataset(self, data: list, threshold: float = 0.7) -> list:
        """
        过滤数据集
        
        Args:
            data: 数据集列表
            threshold: 置信度阈值
            
        Returns:
            过滤后的数据
        """
        filtered = []
        for item in data:
            result = self.judge_sample(item['text'], item['labels'])
            if result['consistent'] and result['confidence'] >= threshold:
                filtered.append(item)
        return filtered


def load_model_and_tokenizer(
    model_name: str = "hfl/chinese-tinybert-l4-uncased",
    num_labels: int = 18,
    **kwargs
) -> Tuple[SentimentAnalyzer, AutoTokenizer]:
    """
    加载模型和分词器
    
    Args:
        model_name: 模型名称或本地路径
        num_labels: 标签数
        **kwargs: 其他参数
        
    Returns:
        (model, tokenizer)
    """
    from pathlib import Path
    
    # 判断是否为本地路径
    is_local = Path(model_name).exists()
    
    if is_local:
        # 本地路径需要转换为绝对路径
        model_path = str(Path(model_name).absolute())
        print(f"从本地加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        base_model_name = model_path  # 用于加载预训练模型权重
    else:
        print(f"从 Hugging Face 加载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model_name = model_name
    
    model = SentimentAnalyzer(
        model_name=base_model_name,
        num_labels=num_labels,
        **kwargs
    )
    
    return model, tokenizer


if __name__ == "__main__":
    # 测试模型
    print("测试模型初始化...")
    model, tokenizer = load_model_and_tokenizer()
    
    # 测试输入
    text = "这个产品真的很不错，性价比很高！"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # 测试前向传播
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    print(f"Logits 形状: {outputs['logits'].shape}")
    print(f"Probabilities 形状: {outputs['probabilities'].shape}")
    print("模型测试通过！")
