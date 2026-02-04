"""
高级损失函数
包含 Focal Loss、Dice Loss 等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    公式: FL = -alpha * (1 - pt)^gamma * log(pt)
    
    对于易分类样本（pt >> 0.5），(1-pt)^gamma 会很小，降低其贡献
    对于难分类样本（pt << 0.5），损失几乎不变
    """
    
    def __init__(self, num_classes=3, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            num_classes: 类别数
            alpha: 类别权重，可以设为 [5.0, 1.0, 2.5] 对应 ASAP
            gamma: 聚焦参数，越大越关注难分类样本
            reduction: 'mean' 或 'sum'
        """
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        else:
            self.alpha = torch.tensor(alpha)
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] logits
            targets: [N] 类别索引
        """
        # 将 alpha 移到与 inputs 相同设备
        alpha = self.alpha.to(inputs.device)
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=alpha)
        
        # 计算 pt (预测正确类别的概率)
        pt = torch.exp(-ce_loss)
        
        # Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss，常用于分割，对类别不平衡很鲁棒
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        # 转换为 one-hot
        num_classes = inputs.size(-1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Softmax 获取概率
        probs = F.softmax(inputs, dim=-1)
        
        # 计算 Dice 系数
        intersection = (probs * targets_one_hot).sum(dim=0)
        union = probs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    组合损失：Focal Loss + Dice Loss
    """
    
    def __init__(self, alpha=None, gamma=2.0, focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs, targets):
        return self.focal_weight * self.focal(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)
