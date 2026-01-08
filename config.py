"""
项目配置文件
"""
import os
from pathlib import Path

# 尝试加载环境变量（dotenv为可选依赖）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # 如果没有安装dotenv，直接从环境变量读取
    pass

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# # API配置（使用OpenAI兼容格式）
# API_KEY = os.getenv("API_KEY", "sk")  # 在 QS 平台生成的 token
# BASE_URL = os.getenv("BASE_URL", "http://10.158.146.63:8081/v1")  # DirectLLM 域名
# MODEL_NAME = os.getenv("MODEL_NAME", "/workspace/models/Qwen3-VL-235B-A22B-Instruct/")  # 模型名称，在 Body 中指明要访问的模型名

# -----------------
# API配置（使用LBOpenAIClient）
SERVICE_NAME = os.getenv("SERVICE_NAME", "mediak8s-editprompt-qwen235b")  # 服务名称
ENV = os.getenv("ENV", "prod")  # 环境（prod/staging等）
API_KEY = os.getenv("API_KEY", "1")  # API密钥（LBOpenAIClient需要，但可能不使用）
MODEL_NAME = os.getenv("MODEL_NAME", "/workspace/Qwen3-VL-235B-A22B-Instruct")  # 模型名称

# 向后兼容配置（已废弃，建议使用上面的配置）
BASE_URL = os.getenv("BASE_URL", "https://maas.devops.xiaohongshu.com/v1")  # DirectLLM 域名（仅用于兼容）
# ------------------

# 向后兼容（已废弃，建议使用上面的配置）
GEMINI_API_KEY = API_KEY
GEMINI_MODEL = MODEL_NAME

# Pipeline配置
PIPELINE_CONFIG = {
    "question": {
        "name": "Question Pipeline",
        "description": "Object recognition + concept matching",
        "question": "Which term matches the picture?",
        "criteria": [
            "Exactly one primary object is present in the image",
            "Primary object exhibits visually identifiable features that correspond to the target concept",
            "Object boundaries are visually clear and separable from the background"
        ]
    },

    "caption": {
        "name": "Caption Pipeline",
        "description": "Scene description/caption",
        "question": "Which one is the correct caption of this image?",
        "criteria": [
            "Image depicts a real-world photographic scene (not illustration, diagram, or synthetic image)",
            "Multiple objects of different semantic types are present",
            "Objects have clearly distinguishable spatial positions and relationships",
            "Objects are contextually consistent with the background environment",
            "Most objects are largely complete and not heavily occluded",
            "[optional] Empty or background-only regions do not exceed approximately one-third of the image"
        ]
    },

    "place_recognition": {
        "name": "Place Recognition Pipeline",
        "description": "Geographic location identification",
        "question": "What is the name of the place shown?",
        "criteria": [
            "Image shows a real and non-fictional geographic location",
            "Map-style image includes a visually highlighted target region",
            "Highlighted region is positioned near the image center",
            "Highlighted region occupies at least half of the image area",
            "Location identity can be inferred from visual information alone without external text"
        ]
    },

    "text_association": {
        "name": "Text Association Pipeline",
        "description": "Image and text correlation",
        "question": "Which can be the associated text with this image posted on twitter?",
        "criteria": [
            "Exactly one primary object is present",
            "Primary object is positioned near the image center",
            "Primary object occupies the majority of the image area (approximately 70–80%)",
            "Primary object is visually salient and unambiguous"
        ]
    },

    "object_proportion": {
        "name": "Object Proportion Pipeline",
        "description": "Object size proportion in image",
        "question": "Approximately what proportion of the picture is occupied by [object]?",
        "criteria": [
            "Image contains at least one visually salient object category that can serve as a potential target, even if not explicitly specified in the query",
            "Target object is complete or sufficiently recognizable for size estimation",
            "Image is not fully dominated by a single object covering nearly the entire frame",
            "Target object boundaries are visually discernible",
            "Target object category is clearly defined without conceptual ambiguity",
            "[optional] Target object may be absent from the image, requiring a zero-proportion judgment"
        ]
    },

    "object_position": {
        "name": "Object Position Pipeline",
        "description": "Object location in image",
        "question": "Where is the [object] located in the picture?",
        "criteria": [
            "Image contains at least one visually salient object that can serve as a potential target",
            "At least one such object has visually identifiable boundaries or distinguishable parts",
            "At least one such object can be treated as a single coherent entity for localization",
            "[optional] Image contains visually similar objects that may act as distractors",
            "[optional] Potential target object occupies a relatively small region of the image",
            "[optional] Potential target object is partially visible (e.g., body parts, silhouette, color patch)",
            "[optional] Potential target object appears in challenging regions (shadows, background, corners, or partial occlusion)"
        ]
    },

    "object_absence": {
        "name": "Object Absence Pipeline",
        "description": "Identifying areas without certain objects",
        "question": "Which corner doesn't have any [objects]?",
        "criteria": [
            "Image contains at least one visually salient object category that can serve as a potential target, even if not explicitly specified in the query",
            "Objects in the image have visually clear boundaries",
            "Objects occupy a substantial portion of the image area",
            "Target object category is clearly defined",
            "Image can be partitioned into distinct and identifiable spatial regions (e.g., four corners)"
        ]
    },

    "object_orientation": {
        "name": "Object Orientation Pipeline",
        "description": "Object facing direction",
        "question": "In the picture, which direction is this [object] facing?",
        "criteria": [
            "Image contains at least one visually salient object that can serve as a target instance, even if the query does not explicitly specify the object",
            "Target object has an identifiable front, head, or directional indicator",
            "Question refers to a single target object instance",
            "[optional] Target object occupies a small portion of the image",
            "[optional] Target object is partially occluded or visually blurred",
            "[optional] Image contains interacting or overlapping objects requiring disambiguation",
            "[optional] Orientation judgment requires fine-grained directional discrimination"
        ]
    },

    "object_counting": {
        "name": "Object Counting Pipeline",
        "description": "Counting objects in image",
        "question": "How many [objects] are in the picture?",
        "criteria": [
            "Image contains at least one visually identifiable object category that can be counted, even if the target category is not explicitly specified in the query",
            "Target objects are countable as discrete instances",
            "[optional] Target objects are partially visible or truncated",
            "[optional] Image contains visually similar distractor objects",
            "[optional] Target objects overlap or occlude each other",
            "[optional] Target objects appear under unusual poses, rotations, or viewpoints",
            "[optional] Target objects visually blend with background colors or textures",
            "[optional] Image contains zero instances of the target object",
            "[optional] Multiple visually similar object subtypes may need to be jointly counted",
            "[optional] Object appearance varies due to packaging, surface decoration, or deformation",
            "[optional] Image exhibits challenging visual conditions (low light, blur, reflections)",
            "[optional] Mirrors or reflections introduce duplicated or misleading object appearances"
        ]
    }
}

# 数据路径配置
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"

# 数据匹配配置
RECAT_ROOT = os.getenv("RECAT_ROOT", "/home/zhuxuzhou/recat")  # 重分类后的根目录
BENCHMARK_FILE = os.getenv("BENCHMARK_FILE", "/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.5/bench/MMBench_DEV_EN_V11/MMBench_DEV_EN_V11.parquet")  # 基准parquet文件路径
MATCH_OUTPUT_DIR = os.getenv("MATCH_OUTPUT_DIR", str(DATA_DIR / "matched_output"))  # 匹配结果输出目录

# 数据匹配测试模式配置
MATCH_TEST_MODE = os.getenv("MATCH_TEST_MODE", "False").lower() == "false"  # 测试模式
MATCH_TEST_SAMPLES = int(os.getenv("MATCH_TEST_SAMPLES", "5"))  # 测试模式下每个类别处理的样本数
MATCH_TEST_MAX_CATEGORIES = int(os.getenv("MATCH_TEST_MAX_CATEGORIES", "2"))  # 测试模式下最多处理的类别数

# 创建必要的目录
DATA_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
Path(MATCH_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

