"""
数据预处理模块
包含文本清洗、Head+Tail 截断策略
适配 ASAP 数据集格式
"""

import re
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from tqdm import tqdm


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, max_length: int = 128, head_ratio: float = 0.6):
        """
        Args:
            max_length: 最大序列长度（token 数，非字符数）
            head_ratio: Head+Tail 策略中头部占比
        """
        self.max_length = max_length
        self.head_ratio = head_ratio
        self.tail_ratio = 1 - head_ratio
        
        # 中文字符平均每个字约 1-2 tokens，预留特殊 token 位置
        self.char_limit = int(max_length * 0.8)
        
    def clean_text(self, text: str) -> str:
        """
        文本清洗
        - 去除多余空白
        - 去除特殊符号（保留中文、英文、数字、常用标点）
        - 规范化空格
        """
        if not isinstance(text, str):
            text = str(text)
            
        # 去除 URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除 email
        text = re.sub(r'\S+@\S+', '', text)
        
        # 去除 @提及 和 #话题#
        text = re.sub(r'@[\w]+', '', text)
        text = re.sub(r'#([^#]+)#', r'\1', text)
        
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 去除重复标点
        text = re.sub(r'([！？。，；：""''（）【】])\1+', r'\1', text)
        
        # 去除非中文、英文、数字、常用标点的字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；：""''（）【】、]', '', text)
        
        return text.strip()
    
    def head_tail_truncate(self, text: str) -> str:
        """
        Head+Tail 拼接截断策略
        保留文本头部（核心语义）和尾部（结论/情感倾向）
        
        Args:
            text: 输入文本
            
        Returns:
            截断后的文本
        """
        text = self.clean_text(text)
        
        # 如果文本较短，直接返回
        if len(text) <= self.char_limit:
            return text
        
        # 计算头部和尾部字符数
        head_len = int(self.char_limit * self.head_ratio)
        tail_len = self.char_limit - head_len
        
        # 截取头部和尾部
        head = text[:head_len]
        tail = text[-tail_len:] if tail_len > 0 else ""
        
        truncated = f"{head}[...]{tail}" if tail else head
        return truncated


def map_asap_labels(value):
    """
    将 ASAP 标签映射到标准格式
    ASAP: -2(未提及), -1(负面), 0(中性), 1(正面)
    映射: -2→1(中性), -1→0(负面), 0→1(中性), 1→2(正面)
    """
    mapping = {
        -2: 1,  # 未提及 → 中性
        -1: 0,  # 负面
        0: 1,   # 中性
        1: 2,   # 正面
    }
    return mapping.get(value, 1)


def process_asap_csv(input_path: str, output_path: str, max_length: int = 128, head_ratio: float = 0.6) -> None:
    """
    处理 ASAP CSV 文件
    
    Args:
        input_path: 输入 CSV 路径
        output_path: 输出 jsonl 路径
        max_length: 最大长度
        head_ratio: Head 比例
    """
    print(f"正在处理 {input_path}...")
    
    preprocessor = TextPreprocessor(max_length=max_length, head_ratio=head_ratio)
    
    # 读取 CSV，自动检测编码
    try:
        df = pd.read_csv(input_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(input_path, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='gb2312')
    
    print(f"总样本数: {len(df)}")
    print(f"所有列名: {df.columns.tolist()}")
    
    # 调试：打印第一行数据
    if len(df) > 0:
        print(f"\n第一行原始数据:")
        first_row = df.iloc[0]
        print(f"  review 类型: {type(first_row.get('review'))}")
        print(f"  review 前50字符: {str(first_row.get('review', ''))[:50]}...")
    
    # ASAP 18 个维度列名（实际的列名格式）
    aspect_columns = [
        'Location#Transportation', 'Location#Downtown', 'Location#Easy_to_find',
        'Service#Queue', 'Service#Hospitality', 'Service#Parking', 'Service#Timely',
        'Price#Level', 'Price#Cost_effective', 'Price#Discount',
        'Ambience#Decoration', 'Ambience#Noise', 'Ambience#Space', 'Ambience#Sanitary',
        'Food#Portion', 'Food#Taste', 'Food#Appearance', 'Food#Recommend'
    ]
    
    processed_data = []
    skip_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="预处理"):
        text = row.get('review', '')
        
        # 处理缺失值
        if pd.isna(text):
            skip_count += 1
            continue
            
        text = str(text).strip()
        
        # 跳过空文本
        if not text or text == 'nan' or len(text) < 3:
            skip_count += 1
            continue
        
        # Head+Tail 截断
        processed_text = preprocessor.head_tail_truncate(text)
        
        # 提取并映射标签
        labels = []
        for col in aspect_columns:
            raw_label = row.get(col, -2)
            # 处理可能的缺失值
            if pd.isna(raw_label):
                raw_label = -2
            mapped_label = map_asap_labels(int(raw_label))
            labels.append(mapped_label)
        
        processed_data.append({
            'text': processed_text,
            'labels': labels,
            'star': int(row.get('star', 3)) if not pd.isna(row.get('star')) else 3,
            'original_length': len(text),
            'processed_length': len(processed_text)
        })
    
    if skip_count > 0:
        print(f"跳过空/无效样本: {skip_count} 条")
    
    if len(processed_data) == 0:
        print("错误：没有成功处理任何样本！")
        print(f"DataFrame 形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"第一行数据:\n{df.iloc[0] if len(df) > 0 else '无数据'}")
        return
    
    # 保存为 jsonl
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n处理完成！")
    print(f"有效样本：{len(processed_data)} 条")
    
    if len(processed_data) == 0:
        print("警告：没有有效样本被处理！")
        return
        
    print(f"输出文件：{output_path}")
    
    # 打印统计信息
    orig_lengths = [item['original_length'] for item in processed_data]
    proc_lengths = [item['processed_length'] for item in processed_data]
    truncated = sum(1 for o, p in zip(orig_lengths, proc_lengths) if p < o)
    
    print(f"\n统计信息：")
    print(f"  原始文本平均长度：{sum(orig_lengths)/len(orig_lengths):.1f}")
    print(f"  处理后平均长度：{sum(proc_lengths)/len(proc_lengths):.1f}")
    print(f"  截断比例：{truncated / len(orig_lengths) * 100:.1f}%")
    
    # 标签分布
    label_counts = {0: 0, 1: 0, 2: 0}
    for item in processed_data:
        for label in item['labels']:
            label_counts[label] += 1
    total = sum(label_counts.values())
    if total > 0:
        print(f"\n标签分布：")
        print(f"  负面(0): {label_counts[0]} ({label_counts[0]/total*100:.1f}%)")
        print(f"  中性(1): {label_counts[1]} ({label_counts[1]/total*100:.1f}%)")
        print(f"  正面(2): {label_counts[2]} ({label_counts[2]/total*100:.1f}%)")


def split_train_eval(train_path: str, eval_path: str, train_output: str, eval_output: str, 
                     eval_ratio: float = 0.1, random_seed: int = 42) -> None:
    """
    从训练集划分出一部分作为验证集（如果 dev.csv 太小）
    
    Args:
        train_path: 训练集路径
        eval_path: 验证集路径（原始 dev）
        train_output: 输出训练集
        eval_output: 输出验证集
        eval_ratio: 从 train 划分的比例
        random_seed: 随机种子
    """
    random.seed(random_seed)
    
    # 读取训练集
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = [json.loads(line) for line in f]
    
    # 随机划分
    random.shuffle(train_data)
    split_idx = int(len(train_data) * (1 - eval_ratio))
    
    final_train = train_data[:split_idx]
    final_eval = train_data[split_idx:]
    
    # 如果原始 dev 比较大，也可以合并
    dev_path = Path(eval_path)
    if dev_path.exists():
        with open(dev_path, 'r', encoding='utf-8') as f:
            dev_data = [json.loads(line) for line in f]
        if len(dev_data) > 1000:
            final_eval = dev_data
            print(f"使用原始 dev 作为验证集: {len(dev_data)} 条")
    
    # 保存
    for path, data in [(train_output, final_train), (eval_output, final_eval)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n数据集划分完成：")
    print(f"  训练集：{len(final_train)} 条 → {train_output}")
    print(f"  验证集：{len(final_eval)} 条 → {eval_output}")


def balance_dataset(input_path: str, output_path: str, strategy: str = 'oversample', target_ratio: dict = None):
    """
    平衡数据集（过采样/欠采样）
    
    ASAP 数据分布极不平衡：负面 4%，中性 76%，正面 20%
    通过过采样（复制少数类）或欠采样（减少多数类）来平衡
    
    Args:
        input_path: 输入 jsonl 路径
        output_path: 输出 jsonl 路径
        strategy: 'oversample'(过采样), 'undersample'(欠采样), 'hybrid'(混合)
        target_ratio: 目标比例 {0: 负面, 1: 中性, 2: 正面}，默认 {0: 0.3, 1: 0.4, 2: 0.3}
    """
    if target_ratio is None:
        # 目标：负面 30%，中性 40%，正面 30%
        target_ratio = {0: 0.3, 1: 0.4, 2: 0.3}
    
    print(f"\n{'='*60}")
    print(f"平衡数据集: {strategy}")
    print(f"目标比例: 负面={target_ratio[0]:.0%}, 中性={target_ratio[1]:.0%}, 正面={target_ratio[2]:.0%}")
    print(f"{'='*60}")
    
    # 读取数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 计算整体情感标签（基于第一个维度"整体"，如果没有则取所有维度的平均）
    def get_overall_label(item):
        labels = item['labels']
        # 如果有18维，使用第一个维度作为整体情感
        if len(labels) > 0:
            return labels[0]  # 使用第一个维度作为代表
        return 1  # 默认中性
    
    # 按标签分类
    label_data = {0: [], 1: [], 2: []}
    for item in data:
        label = get_overall_label(item)
        if label in label_data:
            label_data[label].append(item)
    
    # 统计当前分布
    total = len(data)
    print(f"原始分布:")
    for label, name in [(0, '负面'), (1, '中性'), (2, '正面')]:
        count = len(label_data[label])
        print(f"  {name}({label}): {count} ({count/total:.1%})")
    
    # 计算目标数量
    if strategy == 'oversample':
        # 以最大类为基准进行过采样
        max_count = max(len(label_data[l]) for l in [0, 1, 2])
        target_counts = {l: max_count for l in [0, 1, 2]}
    elif strategy == 'undersample':
        # 以最小类为基准进行欠采样
        min_count = min(len(label_data[l]) for l in [0, 1, 2])
        target_counts = {l: min_count for l in [0, 1, 2]}
    else:  # hybrid 或其他
        # 基于目标比例和总样本数计算
        target_total = sum(len(label_data[l]) for l in [0, 1, 2])
        target_counts = {l: int(target_total * target_ratio[l]) for l in [0, 1, 2]}
    
    # 采样
    balanced_data = []
    for label, name in [(0, '负面'), (1, '中性'), (2, '正面')]:
        samples = label_data[label]
        target = target_counts[label]
        
        if len(samples) < target:
            # 过采样：随机复制
            import random
            random.seed(42)
            sampled = samples + random.choices(samples, k=target - len(samples))
        elif len(samples) > target:
            # 欠采样：随机选择
            import random
            random.seed(42)
            sampled = random.sample(samples, target)
        else:
            sampled = samples
        
        balanced_data.extend(sampled)
        print(f"  {name}: {len(samples)} → {len(sampled)}")
    
    # 打乱顺序
    import random
    random.seed(42)
    random.shuffle(balanced_data)
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n平衡完成: {len(data)} → {len(balanced_data)} 条")
    print(f"输出: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ASAP 数据预处理')
    parser.add_argument('--train_csv', type=str, default='data/raw/train.csv',
                        help='训练集 CSV 路径')
    parser.add_argument('--dev_csv', type=str, default='data/raw/dev.csv',
                        help='验证集 CSV 路径')
    parser.add_argument('--max_length', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--head_ratio', type=float, default=0.6,
                        help='Head+Tail 截断头部比例')
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                        help='从训练集划分验证集的比例')
    parser.add_argument('--balance', type=str, default=None,
                        choices=['oversample', 'undersample', 'hybrid'],
                        help='样本平衡策略（推荐 oversample）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # 处理训练集
    print("="*60)
    print("处理训练集")
    print("="*60)
    train_jsonl = 'data/processed/train_full.jsonl'
    process_asap_csv(args.train_csv, train_jsonl, args.max_length, args.head_ratio)
    
    # 处理验证集
    print("\n" + "="*60)
    print("处理验证集")
    print("="*60)
    dev_jsonl = 'data/processed/dev_full.jsonl'
    process_asap_csv(args.dev_csv, dev_jsonl, args.max_length, args.head_ratio)
    
    # 划分训练/验证
    print("\n" + "="*60)
    print("划分训练/验证集")
    print("="*60)
    split_train_eval(
        train_path=train_jsonl,
        eval_path=dev_jsonl,
        train_output='data/processed/train.jsonl',
        eval_output='data/processed/eval.jsonl',
        eval_ratio=args.eval_ratio
    )
    
    # 样本平衡（可选）
    if args.balance:
        print("\n" + "="*60)
        print("样本平衡处理")
        print("="*60)
        # 备份原始训练集
        import shutil
        shutil.copy('data/processed/train.jsonl', 'data/processed/train_original.jsonl')
        # 进行平衡
        balance_dataset(
            input_path='data/processed/train.jsonl',
            output_path='data/processed/train_balanced.jsonl',
            strategy=args.balance
        )
        # 用平衡后的数据替换
        shutil.copy('data/processed/train_balanced.jsonl', 'data/processed/train.jsonl')
        print("\n已使用平衡后的训练集: data/processed/train.jsonl")
    
    print("\n预处理全部完成！")
    print("输出文件：")
    print("  - data/processed/train.jsonl")
    print("  - data/processed/eval.jsonl")
    if args.balance:
        print("  - data/processed/train_original.jsonl (原始未平衡)")
        print("  - data/processed/train_balanced.jsonl (平衡后)")


if __name__ == "__main__":
    main()
