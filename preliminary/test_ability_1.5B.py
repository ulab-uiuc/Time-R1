import json
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime


# def test_model():
#     # 指定本地模型路径
#     # local_model_path = "/data/zliu331/temporal_reasoning/TinyZero/Qwen/Qwen2.5-3B-Instruct"

#     # 加载模型，强制使用 float32 精度
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     local_model_path,
#     #     torch_dtype=torch.float32,  # 尝试使用 float32
#         # device_map="auto",
#     # )
#     tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
#     model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
#     model.to("cuda:0")  # 指定GPU号9
#     model.eval()  # 切换到评估模式

#     # tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")

#     prompt = (
#                 f"Please carefully read the following news article information:\n"
#                 f"Headline: Dartmouth College Basketball Players Halt Effort to Unionize\n"
#                 f"Abstract: The decision to withdraw the petition appeared to be an effort to preserve a favorable federal ruling that could have been in jeopardy under President-elect Donald J. Trump.\n"
#                 # f"Lead Paragraph: {lead}\n\n"
#                 # "Based on your general knowledge and the details provided, determine whether the article describes an event that has occurred or is anticipated to occur, and infer the specific occurrence date if possible."
#                 "For the purpose of this inference, assume that the event described in the article definitely occurs (or will occur). Based on the information provided and your general knowledge, determine the specific occurrence date of the event."
#                 "- Output the event's occurrence date in the format 'YYYY-MM' on a new line starting with 'Prediction:' (for example, 'Prediction: 2025-03')."
#                 "- Under no circumstances should you output 'Prediction: No event'. Always provide your best inferred date even if the information is ambiguous."
#                 # "IMPORTANT: Do not default to 'No event' without carefully evaluating any temporal clues. Even if the date is not explicitly mentioned but there are hints or context suggesting a timeframe, please provide your best inferred date."
#                 "Your answer must strictly follow the above format and include no additional text."
#             )
#     message_batch = [
#         [{"role": "system", "content": "You are a helpful assistant."},
#          {"role": "user", "content": "Give me a short introduction to large language model."}],
#         [{"role": "system", "content": "You are a helpful assistant."},
#          {"role": "user", "content": prompt}],
#     ]
#     text_batch = tokenizer.apply_chat_template(
#         message_batch,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     model_inputs_batch = tokenizer(
#         text_batch,
#         return_tensors="pt",
#         padding=True
#     ).to(model.device)

#     with torch.no_grad():
#         generated_ids_batch = model.generate(
#             **model_inputs_batch,
#             # temperature=1.0,  # 调整 temperature 值
#             # top_p=0.9,      # 可暂时移除 top_p 参数进行测试
#             max_new_tokens=1024,
#         )

#     new_tokens = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
#     response_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

#     for i, resp in enumerate(response_batch):
#         print(f"Message {i} output:\n{resp}\n")

# if __name__ == "__main__":
#     test_model()





def load_jsonl(file_path):
    """加载 jsonl 格式文件，返回一个列表，每个元素是一个 dict。"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def format_pub_date(pub_date_str):
    """将原始 pub_date 转换为 YYYY-MM 格式（假设 pub_date 为 ISO 格式）。"""
    try:
        dt = datetime.fromisoformat(pub_date_str.rstrip("Z"))
        return dt.strftime("%Y-%m")
    except Exception as e:
        return pub_date_str  # 若解析失败，直接返回原字符串

def extract_prediction(text):
    """
    从文本中提取出以 'Prediction:' 开头的那一行内容。
    如果没有找到，返回 'Prediction: No event'。
    """
    # 先按行分割
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Prediction:"):
            return line
    # 如果没有找到 Prediction:，则返回一个默认值
    return "Prediction: No event"

def batch_inference(
    data_file="NYT_2025_data.jsonl",
    output_file="NYT_2025_prediction.jsonl",
    batch_size=64,
    # local_model_path = "Qwen/Qwen2.5-3B-Instruct"
):
    # 1. 加载模型和分词器，从本地目录或 Hugging Face 上加载
    # 如果你已经将模型下载到本地目录（例如 "Qwen/Qwen2.5-3B-Instruct" 目录下），则直接指定该路径
    
    # model = AutoModelForCausalLM.from_pretrained(
    #     local_model_path,
    #     torch_dtype="auto",
    #     # device_map="auto",  # 自动利用可用 GPU
    # )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model.to("cuda:3")  # 指定GPU号9
    model.eval()  # 切换到评估模式
    # tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    
    # 2. 加载 jsonl 数据
    data = load_jsonl(data_file)
    
    results = []
    
    # 3. 批量推理
    for i in tqdm(range(0, len(data), batch_size), desc="Batch Inference"):
        batch = data[i: i + batch_size]
        message_batch = []
        # 构造 prompt：我们选取 headline, abstract, lead_paragraph
        for record in batch:
            headline = record.get("headline", "")
            abstract = record.get("abstract", "")
            # lead = record.get("lead_paragraph", "")
            # 构造输入 prompt，要求大模型判断是否会发生事件，如果会，输出预测的年份和月份（YYYY-MM），要求统一格式
            # prompt = (
            #     f"Please carefully read the following news article information:\n"
            #     f"Headline: {headline}\n"
            #     f"Abstract: {abstract}\n"
            #     # f"Lead Paragraph: {lead}\n\n"
            #     "Based on your knowledge and the information provided, determine whether the event described in the article has occurred or is predicted to occur."
            #     "- If you conclude that the event is (or will be) occurring, output the event occurrence date in the format 'YYYY-MM' on a new line starting with 'Prediction:' (for example, 'Prediction: 2025-03')."
            #     "- If you determine that there is no event or no specific occurrence date can be inferred, output exactly 'Prediction: No event'."
            #     "Your answer must strictly follow the above format and include no additional text."
            # )
            prompt = (
                f"Please carefully read the following news article information:\n"
                f"Headline: {headline}\n"
                f"Abstract: {abstract}\n"
                # f"Lead Paragraph: {lead}\n\n"
                "Based on your general knowledge and the details provided, determine whether the article describes an event that has occurred or is anticipated to occur, and infer the specific occurrence date if possible."
                "- If the article clearly indicates that an event has occurred or is predicted to occur, output the event's occurrence date in the format 'YYYY-MM' on a new line starting with 'Prediction:' (for example, 'Prediction: 2025-03')."
                "- If the article does not provide sufficient details to infer a specific occurrence date, output exactly 'Prediction: No event'."
                "IMPORTANT: Do not default to 'No event' without carefully evaluating any temporal clues. Even if the date is not explicitly mentioned but there are hints or context suggesting a timeframe, please provide your best inferred date."
                "Your answer must strictly follow the above format and include no additional text."
            )
            # prompt = (
            #     f"Please carefully read the following news article information:\n"
            #     f"Headline: {headline}\n"
            #     f"Abstract: {abstract}\n"
            #     # f"Lead Paragraph: {lead}\n\n"
            #     # "Based on your general knowledge and the details provided, determine whether the article describes an event that has occurred or is anticipated to occur, and infer the specific occurrence date if possible."
            #     "For the purpose of this inference, assume that the event described in the article definitely occurs (or will occur). Based on the information provided and your general knowledge, determine the specific occurrence date of the event."
            #     "- Output the event's occurrence date in the format 'YYYY-MM' on a new line starting with 'Prediction:' (for example, 'Prediction: 2025-03')."
            #     "- Under no circumstances should you output 'Prediction: No event'. Always provide your best inferred date even if the information is ambiguous."
            #     # "IMPORTANT: Do not default to 'No event' without carefully evaluating any temporal clues. Even if the date is not explicitly mentioned but there are hints or context suggesting a timeframe, please provide your best inferred date."
            #     "Your answer must strictly follow the above format and include no additional text."
            # )
            message_batch.append([{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}])
        
        # 利用 Qwen 提供的 chat 模板（如果已在 tokenizer 中配置），转换为文本输入
        text_batch = tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 对文本进行编码（注意根据实际情况设置 max_length 以免截断太多内容）
        model_inputs_batch = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)
        
        # 4. 模型生成预测
        with torch.no_grad():
            generated_ids_batch = model.generate(
                **model_inputs_batch,
                max_new_tokens=1024,  # 输出最大新 token 数量，根据实际需要调整
                # 可添加其他参数，比如 temperature, top_p 等
            )
        
        # 截取生成的部分（去掉 prompt 部分）
        new_tokens = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
        response_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        # 5. 记录每条输入的预测结果和真实日期（pub_date）
        for j, resp in enumerate(response_batch):
            # 将真实日期格式化为 YYYY-MM
            pub_date = batch[j].get("pub_date", "")
            true_date = format_pub_date(pub_date)
            
            # 只提取以 "Prediction:" 开头的行
            prediction_line = extract_prediction(resp.strip())

            result_record = {
                "headline": batch[j].get("headline", ""),
                "abstract": batch[j].get("abstract", ""),
                # "lead_paragraph": batch[j].get("lead_paragraph", ""),
                "true_pub_date": true_date,
                "model_prediction": prediction_line
            }
            results.append(result_record)
    
    # 6. 将结果以 jsonl 格式写入输出文件
    with open(output_file, "w", encoding="utf-8") as fout:
        for record in results:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Batch inference complete. Results saved to {output_file}")

if __name__ == "__main__":
    batch_inference(
        data_file="/data/zliu331/temporal_reasoning/TinyZero/datasets/nyt_years/2025.jsonl",  # 你的输入数据文件（jsonl 格式）
        output_file="/data/zliu331/temporal_reasoning/TinyZero/preliminary/original_ability_result/2025-1-1.5B.jsonl",
        batch_size=200,  # 可根据实际显存情况调整
        # local_model_path = "/data/zliu331/temporal_reasoning/TinyZero/Qwen/Qwen2.5-3B-Instruct"
    )