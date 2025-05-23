# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# def test_model():
# # Specify the local model path
#     local_model_path = "Time-R1/Qwen/Qwen2.5-3B-Instruct"

# # Loading the model
#     model = AutoModelForCausalLM.from_pretrained(
#         local_model_path,
# torch_dtype=torch.float32, #"auto", # Automatic matching accuracy based on graphics card
# device_map="auto", # If using accelerate or transformers 4.30+ device_map function
#     )

# # Load word parter
#     tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")

# # Construct the test input
#     message_batch = [
#         [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role": "user", "content": "Give me a short introduction to large language model."}],
#         [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},{"role": "user", "content": "Give me a summary of LLMs."}],
#     ]
# # A chat template that fits Qwen
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

# # Reasoning
#     with torch.no_grad():
#         generated_ids_batch = model.generate(
#             **model_inputs_batch,
# temperature=1e, # Add temperature coefficient
# # top_p=0.9, # Add top_p sampling
#             max_new_tokens=128,
#         )

# # Decode output
# # Only remove the newly added tokens to avoid decode the propt into it
#     new_tokens = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
#     response_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

# # View the results
#     for i, resp in enumerate(response_batch):
#         print(f"Message {i} output:\n{resp}\n")




# def test_model():
# # Specify the local model path
#     local_model_path = "Time-R1/Qwen/Qwen2.5-3B-Instruct"

# # Load the model and force float32 precision
#     model = AutoModelForCausalLM.from_pretrained(
#         local_model_path,
# torch_dtype=torch.float32, # Try to use float32
#         # device_map="auto",
#     )
# model.to("cuda:9") # Specify GPU number 9
# model.eval() # Switch to evaluation mode

#     tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")

#     message_batch = [
#         [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#          {"role": "user", "content": "Give me a short introduction to large language model."}],
#         [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#          {"role": "user", "content": "Give me a summary of LLMs."}],
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
# # temperature=1.0, # Adjust temperature value
# # top_p=0.9, # The top_p parameter can be temporarily removed for testing
#             max_new_tokens=128,
#         )

#     new_tokens = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
#     response_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

#     for i, resp in enumerate(response_batch):
#         print(f"Message {i} output:\n{resp}\n")

# if __name__ == "__main__":
#     test_model()



import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

def load_jsonl(file_path):
    """Load the jsonl format file and return a list, each element is a dict."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

# def format_pub_date(pub_date_str):
# """Convert the original pub_date to YYYY-MM format (assuming pub_date is ISO format). """
#     try:
#         dt = datetime.fromisoformat(pub_date_str.rstrip("Z"))
#         return dt.strftime("%Y-%m")
#     except Exception as e:
# return pub_date_str # If parsing fails, return to the original string directly

def format_pub_date(pub_date_str):
    """
    Parse the true_pub_date string in the expected format "YYYY-MM-DDTHH:MM:SS+0000".
    Returns a string "YYYY-MM" or None if parsing fails.
    """
    try:
        # If the string ends with 'Z', convert it to "+0000"
        if pub_date_str.endswith("Z"):
            pub_date_str = pub_date_str[:-1] + "+0000"
        # Use strptime to parse strings, note that the time zone format "%z" is suitable for +0000 format
        dt = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%S%z")
    except Exception as e:
        dt = None
    return dt.strftime("%Y-%m") if dt is not None else None

def batch_inference(
    data_file="NYT_2025_data.jsonl",
    output_file="NYT_2025_prediction.jsonl",
    batch_size=64,
    local_model_path = "Qwen/Qwen2.5-3B-Instruct"
):
    # 1. Load the model and word participle, load from the local directory or Hugging Face
    # If you have downloaded the model to a local directory (for example, under the "Qwen/Qwen2.5-3B-Instruct" directory), specify the path directly
    
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype="auto",
        # device_map="auto", # Automatically utilize available GPUs
    )
    model.to("cuda:0")  # Specify GPU number 9
    model.eval()  # Switch to evaluation mode
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, padding_side="left")
    
    # 2. Load jsonl data
    data = load_jsonl(data_file)
    
    results = []
    
    # 3. Batch reasoning
    for i in tqdm(range(0, len(data), batch_size), desc="Batch Inference"):
        batch = data[i: i + batch_size]
        message_batch = []
        # Construct prompt: We select headline, abstract, lead_paragraph
        for record in batch:
            headline = record.get("headline", "")
            abstract = record.get("abstract", "")
            # lead = record.get("lead_paragraph", "")
            # Construct the input prompt, requiring the big model to determine whether an event will occur, if so, output the predicted year and month (YYYY-MM), and require a unified format
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
            # prompt = (
            #     f"Please carefully read the following news article information:\n"
            #     f"Headline: {headline}\n"
            #     f"Abstract: {abstract}\n"
            #     # f"Lead Paragraph: {lead}\n\n"
            #     "Based on your general knowledge and the details provided, determine whether the article describes an event that has occurred or is anticipated to occur, and infer the specific occurrence date if possible."
            #     "- If the article clearly indicates that an event has occurred or is predicted to occur, output the event's occurrence date in the format 'YYYY-MM' on a new line starting with 'Prediction:' (for example, 'Prediction: 2025-03')."
            #     "- If the article does not provide sufficient details to infer a specific occurrence date, output exactly 'Prediction: No event'."
            #     "IMPORTANT: Do not default to 'No event' without carefully evaluating any temporal clues. Even if the date is not explicitly mentioned but there are hints or context suggesting a timeframe, please provide your best inferred date."
            #     "Your answer must strictly follow the above format and include no additional text."
            # )
            prompt = (
                f"Please carefully read the following news article information:\n"
                f"Headline: {headline}\n"
                f"Abstract: {abstract}\n"
                # f"Lead Paragraph: {lead}\n\n"
                # "Based on your general knowledge and the details provided, determine whether the article describes an event that has occurred or is anticipated to occur, and infer the specific occurrence date if possible."
                "For the purpose of this inference, assume that the event described in the article definitely occurs (or will occur). Based on the information provided and your general knowledge, determine the specific occurrence date of the event."
                "- Output the event's occurrence date in the format 'YYYY-MM' on a new line starting with 'Prediction:' (for example, 'Prediction: 2025-03')."
                "- Under no circumstances should you output 'Prediction: No event'. Always provide your best inferred date even if the information is ambiguous."
                # "IMPORTANT: Do not default to 'No event' without carefully evaluating any temporal clues. Even if the date is not explicitly mentioned but there are hints or context suggesting a timeframe, please provide your best inferred date."
                "Your answer must strictly follow the above format and include no additional text."
            )
            message_batch.append([{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}])
        
        # Use the chat template provided by Qwen (if configured in tokenizer) to convert to text input
        text_batch = tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Encode the text (note that max_length is set according to the actual situation to avoid truncating too much content)
        model_inputs_batch = tokenizer(
            text_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(model.device)
        
        # 4. Model Generation Prediction
        with torch.no_grad():
            generated_ids_batch = model.generate(
                **model_inputs_batch,
                max_new_tokens=128,  # Output the maximum number of new tokens, adjust according to actual needs
                # You can add other parameters, such as temperature, top_p, etc.
            )
        
        # Intercept the generated part (remove the prompt part)
        new_tokens = generated_ids_batch[:, model_inputs_batch.input_ids.shape[1]:]
        response_batch = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        # 5. Record the prediction results and real date of each input (pub_date)
        for j, resp in enumerate(response_batch):
            # Format the real date as YYYY-MM
            pub_date = batch[j].get("pub_date", "")
            true_date = format_pub_date(pub_date)
            
            result_record = {
                "headline": batch[j].get("headline", ""),
                "abstract": batch[j].get("abstract", ""),
                # "lead_paragraph": batch[j].get("lead_paragraph", ""),
                "true_pub_date": true_date,
                "model_prediction": resp.strip()  # Predictive text
            }
            results.append(result_record)
    
    # 6. Write the result to the output file in jsonl format
    with open(output_file, "w", encoding="utf-8") as fout:
        for record in results:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Batch inference is completed, and the result is saved in {output_file}")

if __name__ == "__main__":
    batch_inference(
        data_file="Time-R1/datasets/nyt_years/2025.jsonl",  # Your input data file (jsonl format)
        output_file="Time-R1/preliminary/original_ability_result/2025-0.jsonl",
        batch_size=256,  # Can be adjusted according to actual video memory
        local_model_path = "Time-R1/Qwen/Qwen2.5-3B-Instruct"
    )