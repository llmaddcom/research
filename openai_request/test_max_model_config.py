from datasets import load_dataset
from openai import OpenAI

client = OpenAI(
    api_key="sk-proj-1234567890",
    base_url="http://192.168.1.100:8000/v1"
)

# 数据集路径配置
DATASET_BASE_PATH = "/data/disk1/createrool/research/other/qwen3-finetune-test"
FINETOME_PATH = f"{DATASET_BASE_PATH}/FineTome/train/data-00000-of-00001.arrow"
MATH_PATH = f"{DATASET_BASE_PATH}/OpenMathReasoning/cot/data-00000-of-00001.arrow"

def load_conversation_dataset(arrow_file_path, dataset_name):
    """加载对话数据集"""
    print(f"正在加载{dataset_name}数据集...", flush=True)
    dataset = load_dataset("arrow", data_files=arrow_file_path)
    train_dataset = dataset["train"]
    print(f"{dataset_name}数据集加载完成，大小: {len(train_dataset)}", flush=True)
    return train_dataset

# 加载数据集
finetome_dataset = load_conversation_dataset(FINETOME_PATH, "FineTome")
# math_dataset = load_conversation_dataset(MATH_PATH, "OpenMathReasoning")



contents = ""
for i in range(len(finetome_dataset)):
    if len(contents) > 1000000:
        break
    data = finetome_dataset[i]
    # print(f"\n数据 {i+1}:", flush=True)
    if 'conversations' in data:
        for j, conv in enumerate(data['conversations']):
            role = conv.get('from', 'unknown')
            content = conv.get('value', '')
            # print(f"  [{role}]: {content}", flush=True)
            contents += f"[{role}]: {content}"
            print(len(contents))
contents += "模仿这种形式写一些中文案例" 

messages = [{"role": "user", "content": contents}]
print("len(contents):", len(contents))

response = client.chat.completions.create(
    model="Qwen3",
    messages=messages,
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="", flush=True)

        


# for i in range(min(3, len(math_dataset))):
#     data = math_dataset[i]
#     print(f"\n数据 {i+1}:", flush=True)
#     # 打印所有字段
#     for key, value in data.items():
#         if isinstance(value, str) and len(value) > 150:
#             print(f"  {key}: {value[:150]}...", flush=True)
#         else:
#             print(f"  {key}: {value}", flush=True)

# 现在可以遍历数据集了
# 示例：遍历FineTome数据集
# for i, data in enumerate(finetome_dataset):
#     conversations = data.get('conversations', [])
#     source = data.get('source', '')
#     score = data.get('score', 0)
#     # 处理数据...

# 示例：遍历OpenMathReasoning数据集
# for i, data in enumerate(math_dataset):
#     # 处理数据...
