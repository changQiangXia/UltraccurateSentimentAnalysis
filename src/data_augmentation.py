"""
数据增强技术
包含 EDA（Easy Data Augmentation）、回译、改写等
"""

import random
import jieba
import jieba.posseg as pseg
from typing import List


class EDAugmenter:
    """
    Easy Data Augmentation for Chinese
    参考论文: EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks
    """
    
    def __init__(self, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=4):
        """
        Args:
            alpha_sr: 同义词替换比例
            alpha_ri: 随机插入比例
            alpha_rs: 随机交换比例
            p_rd: 随机删除概率
            num_aug: 每个样本生成多少个增强样本
        """
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        self.num_aug = num_aug
        
        # 简单同义词词典（可以扩展）
        self.synonyms = {
            '好': ['棒', '不错', '优秀', '出色'],
            '差': ['糟糕', '不好', '烂', '次'],
            '大': ['巨大', '很大', '宏大'],
            '小': ['微小', '很小', '迷你'],
            '快': ['迅速', '飞快', '快捷'],
            '慢': ['迟缓', '缓慢', '磨蹭'],
            '贵': ['昂贵', '高价', '奢华'],
            '便宜': ['实惠', '廉价', '经济'],
            '喜欢': ['喜爱', '钟爱', '青睐'],
            '讨厌': ['厌恶', '反感', '嫌弃'],
        }
    
    def synonym_replacement(self, text: str, n: int) -> str:
        """同义词替换"""
        words = list(jieba.cut(text))
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word in self.synonyms]))
        
        if len(random_word_list) == 0:
            return text
        
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            if num_replaced >= n:
                break
            synonyms = self.synonyms.get(random_word, [])
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
        
        return ''.join(new_words)
    
    def random_insertion(self, text: str, n: int) -> str:
        """随机插入同义词"""
        words = list(jieba.cut(text))
        new_words = words.copy()
        
        for _ in range(n):
            if not new_words:
                break
            # 找一个有可替换同义词的词
            candidates = [w for w in new_words if w in self.synonyms]
            if not candidates:
                break
            
            add_word = random.choice(candidates)
            add_word_synonyms = self.synonyms[add_word]
            synonym = random.choice(add_word_synonyms)
            
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, synonym)
        
        return ''.join(new_words)
    
    def random_swap(self, text: str, n: int) -> str:
        """随机交换词语"""
        words = list(jieba.cut(text))
        if len(words) < 2:
            return text
        
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ''.join(new_words)
    
    def random_deletion(self, text: str, p: float) -> str:
        """随机删除词语"""
        words = list(jieba.cut(text))
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.uniform(0, 1) > p:
                new_words.append(word)
        
        if not new_words:
            return random.choice(words)
        
        return ''.join(new_words)
    
    def augment(self, text: str) -> List[str]:
        """生成多个增强样本"""
        augmented_texts = []
        
        words = list(jieba.cut(text))
        num_words = len(words)
        
        for _ in range(self.num_aug):
            a = random.randint(0, 3)
            
            if a == 0:
                # 同义词替换
                n_sr = max(1, int(self.alpha_sr * num_words))
                augmented_texts.append(self.synonym_replacement(text, n_sr))
            elif a == 1:
                # 随机插入
                n_ri = max(1, int(self.alpha_ri * num_words))
                augmented_texts.append(self.random_insertion(text, n_ri))
            elif a == 2:
                # 随机交换
                n_rs = max(1, int(self.alpha_rs * num_words))
                augmented_texts.append(self.random_swap(text, n_rs))
            else:
                # 随机删除
                augmented_texts.append(self.random_deletion(text, self.p_rd))
        
        return augmented_texts


def augment_dataset_with_eda(input_path: str, output_path: str, target_size: int = None):
    """
    使用 EDA 增强数据集
    
    Args:
        input_path: 输入 jsonl
        output_path: 输出 jsonl
        target_size: 目标样本数，如果为 None 则每个样本生成 4 个增强样本
    """
    import json
    from pathlib import Path
    from tqdm import tqdm
    
    print(f"加载数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"原始样本数: {len(data)}")
    
    augmenter = EDAugmenter(num_aug=2)  # 每个样本生成 2 个增强样本
    
    augmented_data = data.copy()
    
    for item in tqdm(data, desc="EDA 增强"):
        text = item['text']
        labels = item['labels']
        
        # 生成增强样本
        aug_texts = augmenter.augment(text)
        
        for aug_text in aug_texts:
            if aug_text != text:  # 避免重复
                augmented_data.append({
                    'text': aug_text,
                    'labels': labels.copy(),
                    'augmented': True,
                    'original_text': text
                })
        
        # 如果设置了目标大小，提前停止
        if target_size and len(augmented_data) >= target_size:
            break
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in augmented_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"增强完成: {len(data)} → {len(augmented_data)} 条")
    print(f"输出: {output_path}")


if __name__ == "__main__":
    # 测试
    augmenter = EDAugmenter()
    text = "这个产品真的很好，性价比很高！"
    print(f"原文: {text}")
    print("增强:")
    for i, aug in enumerate(augmenter.augment(text)):
        print(f"  {i+1}. {aug}")
