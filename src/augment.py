"""
LLM 辅助数据增强模块
用于生成解释性伪标签和清洗噪声数据
注意：需要配置 API Key 或本地模型
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class AugmentedSample:
    """增强样本"""
    original_text: str
    augmented_text: str
    labels: List[int]
    explanation: str
    confidence: float
    augmentation_type: str


class LLMAugmenter:
    """
    LLM 数据增强器
    支持 Qwen、GPT、Claude 等模型
    """
    
    def __init__(
        self,
        model_name: str = "qwen2-7b-instruct",
        use_api: bool = False,
        api_key: str = None,
        base_url: str = None,
        device: str = "cuda"
    ):
        """
        Args:
            model_name: 模型名称
            use_api: 是否使用 API
            api_key: API Key
            base_url: API Base URL
            device: 本地模型设备
        """
        self.model_name = model_name
        self.use_api = use_api
        self.device = device
        
        if use_api:
            self.api_key = api_key or os.getenv('LLM_API_KEY')
            self.base_url = base_url or os.getenv('LLM_BASE_URL')
        else:
            # 本地加载模型（需要足够的显存）
            self._load_local_model()
    
    def _load_local_model(self):
        """加载本地模型"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"正在加载本地模型：{self.model_name}")
            
            # 使用 4-bit 量化加载（节省显存）
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            print("模型加载完成！")
            
        except ImportError:
            print("请安装必要的包：pip install transformers accelerate bitsandbytes")
            raise
    
    def _call_api(self, prompt: str) -> str:
        """调用 API"""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 512
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        return response.json()['choices'][0]['message']['content']
    
    def _call_local(self, prompt: str) -> str:
        """调用本地模型"""
        import torch
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        return response
    
    def generate_explanation(self, text: str, labels: List[int]) -> str:
        """
        生成情感解释
        
        Args:
            text: 评论文本
            labels: 18维标签
            
        Returns:
            解释文本
        """
        prompt = f"""请分析以下评论的情感倾向，并给出详细的判断依据。

评论内容：
{text}

情感标签（18个维度，0=负面，1=中性，2=正面）：
{labels}

请按以下格式输出：
1. 整体情感分析
2. 各维度详细解读
3. 关键情感词汇提取
4. 判断依据总结
"""
        
        if self.use_api:
            return self._call_api(prompt)
        else:
            return self._call_local(prompt)
    
    def rephrase_text(self, text: str, labels: List[int]) -> str:
        """
        改写文本（保持情感不变）
        
        Args:
            text: 原文本
            labels: 原标签
            
        Returns:
            改写后的文本
        """
        prompt = f"""请用不同的表达方式改写以下评论，保持原意和情感倾向不变。

原评论：
{text}

改写要求：
1. 使用不同的词汇和句式
2. 保持原有的情感强度和倾向
3. 不改变核心观点和评价对象
4. 输出只需要改写后的文本，不要其他解释

改写结果：
"""
        
        if self.use_api:
            return self._call_api(prompt)
        else:
            return self._call_local(prompt)
    
    def judge_sample_quality(self, text: str, labels: List[int]) -> Dict:
        """
        评估样本质量（LLM-as-a-Judge）
        
        Args:
            text: 评论文本
            labels: 情感标签
            
        Returns:
            质量评估结果
        """
        prompt = f"""请评估以下评论与其情感标签的匹配程度。

评论：
{text}

标签（18维，0=负面，1=中性，2=正面）：
{labels}

请按以下 JSON 格式返回评估结果：
{{
    "consistent": true/false,  // 标签是否与评论一致
    "confidence": 0.0-1.0,     // 置信度
    "suggested_labels": [],    // 如果不一致，建议的正确标签
    "reason": "..."            // 评估理由
}}
"""
        
        try:
            if self.use_api:
                response = self._call_api(prompt)
            else:
                response = self._call_local(prompt)
            
            # 尝试解析 JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {
                    "consistent": True,
                    "confidence": 0.5,
                    "reason": "无法解析响应"
                }
                
        except Exception as e:
            return {
                "consistent": True,
                "confidence": 0.5,
                "reason": f"评估出错: {str(e)}"
            }
    
    def augment_dataset(
        self,
        input_path: str,
        output_path: str,
        augmentation_types: List[str] = None,
        max_samples: int = None
    ):
        """
        增强数据集
        
        Args:
            input_path: 输入数据路径
            output_path: 输出路径
            augmentation_types: 增强类型列表
            max_samples: 最大处理样本数
        """
        if augmentation_types is None:
            augmentation_types = ['explanation', 'rephrase']
        
        # 加载数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        if max_samples:
            data = data[:max_samples]
        
        augmented_data = []
        
        for i, item in enumerate(data):
            print(f"处理样本 {i+1}/{len(data)}...")
            
            text = item['text']
            labels = item['labels']
            
            # 保留原始样本
            augmented_data.append(item)
            
            # 添加解释
            if 'explanation' in augmentation_types:
                try:
                    explanation = self.generate_explanation(text, labels)
                    augmented_data.append({
                        'text': text,
                        'labels': labels,
                        'explanation': explanation,
                        'type': 'with_explanation'
                    })
                except Exception as e:
                    print(f"  生成解释失败: {e}")
            
            # 改写文本
            if 'rephrase' in augmentation_types:
                try:
                    rephrased = self.rephrase_text(text, labels)
                    augmented_data.append({
                        'text': rephrased,
                        'labels': labels,
                        'original_text': text,
                        'type': 'rephrased'
                    })
                except Exception as e:
                    print(f"  改写失败: {e}")
        
        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in augmented_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n增强完成！")
        print(f"原始样本数：{len(data)}")
        print(f"增强后样本数：{len(augmented_data)}")
        print(f"输出文件：{output_path}")
    
    def filter_noisy_data(
        self,
        input_path: str,
        output_path: str,
        confidence_threshold: float = 0.7
    ):
        """
        过滤噪声数据
        
        Args:
            input_path: 输入数据路径
            output_path: 输出路径
            confidence_threshold: 置信度阈值
        """
        # 加载数据
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        filtered_data = []
        
        for i, item in enumerate(data):
            print(f"评估样本 {i+1}/{len(data)}...")
            
            result = self.judge_sample_quality(item['text'], item['labels'])
            
            if result['consistent'] and result['confidence'] >= confidence_threshold:
                item['judge_result'] = result
                filtered_data.append(item)
            else:
                print(f"  过滤样本：{result['reason']}")
        
        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in filtered_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n过滤完成！")
        print(f"原始样本数：{len(data)}")
        print(f"过滤后样本数：{len(filtered_data)}")
        print(f"过滤比例：{(1 - len(filtered_data)/len(data)) * 100:.1f}%")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LLM 数据增强')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['augment', 'filter'],
                        help='模式：augment（增强）或 filter（过滤）')
    parser.add_argument('--input_path', type=str, required=True,
                        help='输入数据路径')
    parser.add_argument('--output_path', type=str, required=True,
                        help='输出数据路径')
    parser.add_argument('--model_name', type=str, default='qwen2-7b-instruct',
                        help='模型名称')
    parser.add_argument('--use_api', action='store_true',
                        help='使用 API 而非本地模型')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大处理样本数')
    
    args = parser.parse_args()
    
    # 创建增强器
    augmenter = LLMAugmenter(
        model_name=args.model_name,
        use_api=args.use_api
    )
    
    # 执行增强或过滤
    if args.mode == 'augment':
        augmenter.augment_dataset(
            input_path=args.input_path,
            output_path=args.output_path,
            max_samples=args.max_samples
        )
    else:
        augmenter.filter_noisy_data(
            input_path=args.input_path,
            output_path=args.output_path
        )


if __name__ == "__main__":
    main()
