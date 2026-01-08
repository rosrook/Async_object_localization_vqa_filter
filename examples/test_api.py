import os
from PIL import Image
import io,sys
import base64
from io import BytesIO
import requests

# 替换为你的服务器IP
SERVER_IP = "10.146.229.234"  # 改成你的实际IP
# SERVER_IP = "10.158.146.63"  # 改成你的实际IP
PORT = 22002
# PORT = 8081


# 调用API
api_url = f"http://{SERVER_IP}:{PORT}/v1/chat/completions"

import openai
from PIL import Image
import io
import base64


# http://10.158.146.63:8081/v1/chat/completions


client = openai.OpenAI(
    api_key="sk",
    base_url=f"http://10.158.146.63:8081/v1",
    timeout=60,
)


def get_api_result(text, img_base64):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "{}".format(text)},
            ],
        }
    ]
    if isinstance(img_base64, list):
        for img_b64 in img_base64:
            img_cont = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}"
                    }
                 }
            messages[0]['content'].append(img_cont)
    else:
        img_cont = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_base64}"
            }
        }
        messages[0]['content'].append(img_cont)

    response = client.chat.completions.create(
        model="/workspace/models/Qwen3-VL-235B-A22B-Instruct/",
        messages=messages,
        # extra_body={
        #     "thinking_level": "low",
        #     "include_thoughts": True
        # },
        # temperature=0.6,
        max_tokens=26000,
    )

    response_text = response.choices[0].message.content
    print("响应:", response_text)
    return response_text



def chat_template(question, image_base64):
    res =  {
        "model": "default",
        "temperature": 0.7,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                 "content": [
                 {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                 },
                 {
                    "type": "text",
                    "text": question
                 }
                ]
        }
        ]
    }

    # if isinstance(image_base64, list):
    #     for img_b64 in image_base64:
    #         img_cont = {
    #                 "type": "image_url",
    #                 "image_url": {
    #                     "url": f"data:image/jpeg;base64,{img_b64}"
    #                 }
    #             }
    #         res['messages'][0]['content'].append(img_cont)
    # else:
    #     img_cont = {
    #         "type": "image_url",
    #         "image_url": {
    #             "url": f"data:image/jpeg;base64,{image_base64}"
    #         }
    #     }
    #     res['messages'][0]['content'].append(img_cont)
    # res['messages'][0]['content'].append(
    #     {
    #         "type": "text",
    #         "text": question
    #     }
    # )
    return res

# def get_api_result(payload):
#     response = requests.post(api_url, json=payload)
#     response_text = response.json()['choices'][0]['message']['content']
#     return response_text


def to_base64(img):
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, bytes):
        img = Image.open(io.BytesIO(img)).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    test_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return test_image_base64


def test_local_model():
    try:
        # 测试文本
        test_text = "这是一个测试问题"

        # 测试图片，可以用本地一张小图片
        test_image_path = "test.jpg"
        if not os.path.exists(test_image_path):
            # 如果没有图片，生成一张纯白小图
            img = Image.new("RGB", (64, 64), color=(255, 255, 255))
            img.save(test_image_path)

        # 转 base64
        img_base64 = to_base64(test_image_path)

        # 调用模型
        print("正在调用本地模型服务...")
        response_text = get_api_result(test_text, img_base64)
        print("\n模型返回结果:")
        print(response_text)

    except Exception as e:
        print("调用失败:", str(e))

if __name__ == "__main__":
    test_local_model()
