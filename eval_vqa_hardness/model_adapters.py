"""
不同VQA模型的适配器
根据你的具体模型类型选择或自定义
"""

import torch
from PIL import Image
from typing import Optional


class BaseVQAAdapter:
    """VQA模型适配器基类"""
    
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
    
    def load_model(self):
        """加载模型（需要子类实现）"""
        raise NotImplementedError
    
    def prepare_inputs(self, image: Image.Image, question: str):
        """准备模型输入（需要子类实现）"""
        raise NotImplementedError
    
    def generate_answer(self, inputs, max_new_tokens: int = 128):
        """生成答案（需要子类实现）"""
        raise NotImplementedError


class BLIPAdapter(BaseVQAAdapter):
    """BLIP/BLIP-2模型适配器"""
    
    def load_model(self):
        from transformers import BlipProcessor, BlipForQuestionAnswering
        # 或 Blip2Processor, Blip2ForConditionalGeneration
        
        print("加载BLIP模型...")
        self.processor = BlipProcessor.from_pretrained(self.model_path)
        self.model = BlipForQuestionAnswering.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
    
    def prepare_inputs(self, image: Image.Image, question: str):
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device)
        return inputs
    
    def generate_answer(self, inputs, max_new_tokens: int = 128):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_new_tokens
            )
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class LLaVAAdapter(BaseVQAAdapter):
    """LLaVA模型适配器"""
    
    def load_model(self):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        print("加载LLaVA模型...")
        self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
    
    def prepare_inputs(self, image: Image.Image, question: str):
        # LLaVA通常使用对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs
    
    def generate_answer(self, inputs, max_new_tokens: int = 128):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        # 解码并移除输入部分
        full_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        # 提取assistant的回答
        if "ASSISTANT:" in full_text:
            answer = full_text.split("ASSISTANT:")[-1].strip()
        else:
            answer = full_text
        
        return answer


class InstructBLIPAdapter(BaseVQAAdapter):
    """InstructBLIP模型适配器"""
    
    def load_model(self):
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        
        print("加载InstructBLIP模型...")
        self.processor = InstructBlipProcessor.from_pretrained(self.model_path)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()
    
    def prepare_inputs(self, image: Image.Image, question: str):
        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        ).to(self.device)
        return inputs
    
    def generate_answer(self, inputs, max_new_tokens: int = 128):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=5
            )
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]


class Qwen2VLAdapter(BaseVQAAdapter):
    """Qwen2-VL模型适配器"""
    
    def load_model(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        print("加载Qwen2-VL模型...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
    
    def prepare_inputs(self, image: Image.Image, question: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        return inputs
    
    def generate_answer(self, inputs, max_new_tokens: int = 128):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )
        
        # 只解码新生成的tokens
        generated_ids = outputs[:, inputs['input_ids'].shape[1]:]
        answer = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return answer


class CogVLMAdapter(BaseVQAAdapter):
    """CogVLM模型适配器"""
    
    def load_model(self):
        from transformers import AutoModelForCausalLM, LlamaTokenizer
        
        print("加载CogVLM模型...")
        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
    
    def prepare_inputs(self, image: Image.Image, question: str):
        # CogVLM特定的输入格式
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=question,
            history=[],
            images=[image]
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
    def generate_answer(self, inputs, max_new_tokens: int = 128):
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def get_adapter(model_type: str, model_path: str, device: str) -> BaseVQAAdapter:
    """
    根据模型类型获取对应的适配器
    
    Args:
        model_type: 模型类型 (blip, llava, instructblip, qwen2vl, cogvlm, custom)
        model_path: 模型路径
        device: 设备
        
    Returns:
        对应的适配器实例
    """
    adapters = {
        'blip': BLIPAdapter,
        'llava': LLaVAAdapter,
        'instructblip': InstructBLIPAdapter,
        'qwen2vl': Qwen2VLAdapter,
        'cogvlm': CogVLMAdapter,
    }
    
    adapter_class = adapters.get(model_type.lower())
    if adapter_class is None:
        raise ValueError(
            f"不支持的模型类型: {model_type}. "
            f"支持的类型: {list(adapters.keys())}"
        )
    
    adapter = adapter_class(model_path, device)
    adapter.load_model()
    
    return adapter


# 使用示例
if __name__ == "__main__":
    # 示例：加载BLIP模型
    adapter = get_adapter(
        model_type="blip",
        model_path="Salesforce/blip-vqa-base",
        device="cuda"
    )
    
    # 测试
    from PIL import Image
    image = Image.open("test.jpg")
    question = "What is in the image?"
    
    inputs = adapter.prepare_inputs(image, question)
    answer = adapter.generate_answer(inputs)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")