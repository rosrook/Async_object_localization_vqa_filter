# 异步调用模型完整流程说明

## 调用流程图

```
pipeline.py (main)
    ↓
pipeline.run() [line 565]
    ↓
_generate_answers_from_data_async() [line 438] (如果 concurrency > 1)
    ↓
AnswerGenerator.generate_answer_async() [answer_generator.py:85]
    ↓
├─ _generate_multiple_choice_answer_async() [answer_generator.py:182]
│   ├─ _generate_correct_answer_async() [answer_generator.py:352]
│   │   └─ AsyncGeminiClient.analyze_image_async() [async_client.py:196]
│   └─ _generate_wrong_options_async() [answer_generator.py:536]
│       └─ AsyncGeminiClient.analyze_image_async() [async_client.py:196]
│
└─ _generate_fill_in_blank_answer_async() [answer_generator.py:272]
    └─ _generate_correct_answer_async() [answer_generator.py:352]
        └─ AsyncGeminiClient.analyze_image_async() [async_client.py:196]
```

## 详细代码位置和说明

### 1. 入口点：pipeline.py

**文件位置**: `generate_vqa/pipeline.py`

#### 1.1 main() 函数 [line 1059]
- **作用**: 命令行入口，解析参数
- **关键参数**: `--concurrency` (line 1126-1130)
  - 默认值: 1 (串行)
  - >1 时启用异步并发

#### 1.2 pipeline.run() 方法 [line 565]
- **作用**: 主流程控制
- **关键代码**:
  ```python
  # Line 708-716: 根据并发设置选择同步或异步
  if concurrency and concurrency > 1:
      batch_answers, answer_errors = asyncio.run(
          self._generate_answers_from_data_async(
              batch_questions,
              concurrency=concurrency
          )
      )
  else:
      batch_answers, answer_errors = self._generate_answers_from_data(batch_questions)
  ```

#### 1.3 _generate_answers_from_data_async() 方法 [line 438]
- **作用**: 异步批量生成答案
- **关键逻辑**:
  - **Line 449**: 创建 `AsyncGeminiClient` 上下文管理器
    ```python
    async with AsyncGeminiClient(max_concurrent=concurrency) as client:
    ```
  - **Line 450-549**: 定义 `process_one()` 异步函数，处理单个问题
    - 调用 `answer_generator.generate_answer_async()` (line 488)
    - 传入 `async_client=client` 参数
  - **Line 551-552**: 创建所有异步任务并并发执行
    ```python
    tasks = [process_one(i, rec) for i, rec in enumerate(questions_data, 1)]
    results_raw = await asyncio.gather(*tasks, return_exceptions=False)
    ```

---

### 2. 答案生成层：answer_generator.py

**文件位置**: `generate_vqa/generate_answer/answer_generator.py`

#### 2.1 generate_answer_async() 方法 [line 85]
- **作用**: 异步答案生成入口
- **参数**:
  - `question`: 问题文本
  - `image_base64`: 图片base64编码
  - `question_type`: 题型 ("multiple_choice" 或 "fill_in_blank")
  - `async_client`: `AsyncGeminiClient` 实例（可选）
- **路由逻辑**:
  - `multiple_choice` → `_generate_multiple_choice_answer_async()` (line 96)
  - `fill_in_blank` → `_generate_fill_in_blank_answer_async()` (line 103)

#### 2.2 _generate_multiple_choice_answer_async() 方法 [line 182]
- **作用**: 异步生成选择题答案
- **流程**:
  1. **Line 192-197**: 调用 `_generate_correct_answer_async()` 生成正确答案
  2. **Line 210-216**: 调用 `_generate_wrong_options_async()` 生成错误选项
  3. **Line 227-241**: 组合选项、打乱顺序、确定正确答案字母

#### 2.3 _generate_correct_answer_async() 方法 [line 352]
- **作用**: 异步生成正确答案
- **关键代码**:
  ```python
  # Line 378-391: 调用异步API
  if async_client is None:
      async with AsyncGeminiClient() as client:
          response = await client.analyze_image_async(
              image_input=image_base64,
              prompt=prompt,
              temperature=self.gen_settings.get("temperature", 0.7),
          )
  else:
      response = await async_client.analyze_image_async(
          image_input=image_base64,
          prompt=prompt,
          temperature=self.gen_settings.get("temperature", 0.7),
      )
  ```
- **错误处理**: Line 395-408
  - 特别处理 400 Bad Request 错误
  - 输出详细的诊断信息

#### 2.4 _generate_wrong_options_async() 方法 [line 536]
- **作用**: 异步生成错误选项
- **调用方式**: 与 `_generate_correct_answer_async()` 类似
- **Line 573-585**: 调用 `async_client.analyze_image_async()`

#### 2.5 _generate_fill_in_blank_answer_async() 方法 [line 272]
- **作用**: 异步生成填空题答案
- **流程**: 直接调用 `_generate_correct_answer_async()` (line 282)

---

### 3. API 客户端层：async_client.py

**文件位置**: `utils/async_client.py`

#### 3.1 AsyncGeminiClient 类 [line 17]
- **作用**: 异步API客户端，支持高并发

#### 3.2 __init__() 方法 [line 20]
- **关键参数**:
  - `api_key`: API密钥（从 config.API_KEY 读取）
  - `model_name`: 模型名称（从 config.MODEL_NAME 读取）
  - `base_url`: API基础URL（从 config.BASE_URL 读取）
  - `max_concurrent`: 最大并发数（默认10）
- **Line 39-47**: API Key 验证
  - 检查是否为默认值 "1"
  - 如果无效，抛出清晰的错误提示
- **Line 59**: 创建信号量控制并发
  ```python
  self.semaphore = asyncio.Semaphore(max_concurrent)
  ```

#### 3.3 __aenter__() 方法 [line 61]
- **作用**: 异步上下文管理器入口
- **Line 63-66**: 设置请求头
  ```python
  headers = {
      "Authorization": f"Bearer {self.api_key}",
      "Content-Type": "application/json"
  }
  ```
- **Line 68-71**: 创建 aiohttp.ClientSession

#### 3.4 _normalize_model_name() 方法 [line 168]
- **作用**: 规范化模型名称
- **处理逻辑**:
  1. 移除路径前缀（如 `/workspace/`）
  2. 转换为全小写
  3. 移除末尾斜杠
- **示例**: `/workspace/Qwen3-VL-235B-A22B-Instruct` → `qwen3-vl-235b-a22b-instruct`

#### 3.5 _encode_image() 方法 [line 79]
- **作用**: 编码图片为base64，自动压缩过大图片
- **压缩策略**:
  1. **Line 93-97**: 如果最大边长 > 2048px，自动缩放
  2. **Line 100-115**: 如果base64 > 7MB，降低JPEG质量（从85逐步降到50）
  3. **Line 119-123**: 最终检查，如果仍 > 10MB，抛出错误

#### 3.6 analyze_image_async() 方法 [line 196]
- **作用**: 异步分析图片（核心API调用方法）
- **流程**:
  1. **Line 213**: 使用信号量限制并发
     ```python
     async with self.semaphore:  # 限制并发数
     ```
  2. **Line 218**: 编码图片（自动压缩）
  3. **Line 221**: 规范化模型名称
  4. **Line 224-244**: 构建请求payload
     ```python
     url = f"{self.base_url}/chat/completions"
     payload = {
         "model": normalized_model_name,
         "messages": [
             {
                 "role": "user",
                 "content": [
                     {"type": "text", "text": prompt},
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": f"data:image/jpeg;base64,{image_base64}"
                         }
                     }
                 ]
             }
         ],
         "stream": False,
         "max_tokens": 4096,
         "temperature": temperature
     }
     ```
  5. **Line 248**: 发送POST请求
     ```python
     async with self.session.post(url, json=payload) as response:
     ```
  6. **Line 250-267**: 错误处理
     - 检查HTTP状态码
     - 400错误：尝试解析JSON错误信息
     - 抛出 `aiohttp.ClientResponseError`
  7. **Line 269-275**: 解析响应
     - 提取 `result["choices"][0]["message"]["content"]`
  8. **Line 276-294**: 异常处理和错误信息增强
     - 400错误：显示请求参数、图片大小、模型名称等诊断信息
     - 401错误：显示API Key相关诊断信息

---

## 关键配置

### config.py
- **API_KEY**: API密钥（环境变量或默认值 "1"）
- **MODEL_NAME**: 模型名称（默认: `/workspace/Qwen3-VL-235B-A22B-Instruct`）
- **BASE_URL**: API基础URL（默认: `https://maas.devops.xiaohongshu.com/v1`）

---

## 并发控制机制

1. **信号量控制** (`asyncio.Semaphore`)
   - 位置: `async_client.py:59`
   - 作用: 限制同时进行的API请求数量
   - 默认值: 10（可通过 `max_concurrent` 参数调整）

2. **任务并发** (`asyncio.gather`)
   - 位置: `pipeline.py:552`
   - 作用: 并发执行所有问题的答案生成任务
   - 注意: 实际并发数受信号量限制

---

## 错误处理机制

### 1. API Key 验证
- **位置**: `async_client.py:39-47`
- **检查**: 是否为默认值 "1" 或空字符串
- **错误提示**: 指导用户设置正确的API Key

### 2. 400 Bad Request 错误
- **位置**: `async_client.py:254-260, 279-286`
- **处理**: 
  - 尝试解析JSON错误信息
  - 显示请求URL、模型名称、图片大小等诊断信息

### 3. 401 认证错误
- **位置**: `async_client.py:287-293`
- **处理**: 
  - 显示API Key（部分，保护隐私）
  - 提供设置指导

### 4. 图片压缩
- **位置**: `async_client.py:79-131`
- **策略**: 
  - 自动缩放过大图片
  - 自动降低质量以减小base64大小
  - 防止请求体过大导致400错误

---

## 使用示例

```python
# 启用异步并发（并发数=10）
python generate_vqa/pipeline.py input.json output_dir/ --concurrency 10

# 串行处理（默认）
python generate_vqa/pipeline.py input.json output_dir/
```

---

## 性能优化点

1. **图片自动压缩**: 避免base64过大导致400错误
2. **模型名称规范化**: 自动处理路径前缀和大小写
3. **并发控制**: 通过信号量避免过载
4. **错误重试**: 在pipeline层实现（最多3次重试）
5. **批量处理**: 支持大文件分批处理（batch_size参数）

---

## 常见问题排查

### 1. 401 认证错误
- **检查**: API Key是否正确设置
- **位置**: `async_client.py:287-293` 查看详细错误信息

### 2. 400 Bad Request
- **检查**: 
  - 图片base64是否过大（应 < 10MB）
  - 模型名称格式是否正确
  - 请求参数格式是否正确
- **位置**: `async_client.py:279-286` 查看详细诊断信息

### 3. 模型名称错误
- **检查**: 配置的模型名称是否在API支持列表中
- **处理**: `_normalize_model_name()` 会自动规范化
- **位置**: `async_client.py:168-194`

