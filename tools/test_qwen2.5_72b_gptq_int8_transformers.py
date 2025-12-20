# 文件名：test_qwen2.5_72b_gptq_int8.py
from vllm import LLM, SamplingParams
import os

# 1. 配置缓存目录（避免旧缓存干扰，可选配置）
os.environ["VLLM_CACHE_DIR"] = "/tmp/vllm_cache"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_hub_cache"

# 2. 模型路径（与你的实际模型目录完全一致，无需修改）
model_path = "/data/model/Qwen2.5-72B-Instruct-GPTQ-Int8"

# 3. 初始化 vLLM 模型（核心配置：适配 8×4090，优化显存与速度）
llm = LLM(
    model=model_path,
    tensor_parallel_size=8,  # 关键：使用8卡并行（你的硬件是8×4090，分摊显存压力）
    quantization="gptq",     # 关键：对应模型的 GPTQ-Int8 量化格式，自动识别配置
    gpu_memory_utilization=0.9,  # 最大化利用显存（预留10%冗余，避免4090显存溢出）
    max_model_len=4096,      # 控制KV缓存大小，可根据需求调整（如2048/8192，调低更省显存）
    enforce_eager=True,      # 消费级显卡（4090）必备，避免编译兼容报错
    trust_remote_code=True,  # Qwen 系列模型必需，加载自定义模型代码
    dtype="float16",         # 适配量化模型，提升推理效率
    swap_space=4,            # 预留少量交换空间，防止显存临时不足
)

# 4. 配置生成参数（可根据测试需求灵活调整）
sampling_params = SamplingParams(
    temperature=0.7,        # 随机性控制：0=确定性输出，1=最大随机性，0.7兼顾多样与稳定
    top_p=0.95,             # 核采样：过滤低概率词汇，提升输出质量
    max_tokens=2048,        # 最大生成长度：可调整为512/1024/4096
    stop_token_ids=[151645],# Qwen 模型专属 EOS token ID，确保生成正常终止
    skip_special_tokens=True # 跳过特殊令牌（如<|im_start|>），输出更整洁
)

# 5. 测试提示词（覆盖不同任务类型，验证模型核心能力）
test_prompts = [
    "请简要介绍 Qwen2.5-72B-Instruct-GPTQ-Int8 模型的核心优势、量化特点及适用场景",
    "求解数学应用题：小明从家到学校，步行速度是每分钟60米，需要15分钟到达；若骑自行车每分钟行150米，提前几分钟能到学校？",
    "用 Python 实现一个简单的冒泡排序算法，要求附带详细注释和测试案例",
    "请总结人工智能在金融领域的3个核心应用场景，并分析每个场景的优势与潜在风险"
]

# 6. 执行批量推理（vLLM 批量处理效率更高）
print("===== 开始运行 Qwen2.5-72B-Instruct-GPTQ-Int8 模型推理 =====")
print(f"模型路径：{model_path}")
print(f"测试任务数量：{len(test_prompts)}")
print("-" * 80)

# 批量生成输出
outputs = llm.generate(test_prompts, sampling_params)

# 7. 格式化打印推理结果
for idx, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\n===== 测试任务 {idx+1} =====")
    print(f"【输入提示词】：{prompt}")
    print(f"【模型输出】：{generated_text}")
    print("-" * 80)

print("===== 所有测试任务推理完成 =====")
