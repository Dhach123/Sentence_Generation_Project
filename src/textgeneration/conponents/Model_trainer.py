from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import pandas as pd
from textgeneration.entity import ModelTrainerConfig
from pathlib import Path



def load_dataset(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def load_tokenizer(model_ckpt: str):
    return AutoTokenizer.from_pretrained(model_ckpt)

def load_model(model_ckpt: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        device_map={"": device},
        load_in_8bit=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    if torch.__version__ >= "2.0":
        model = torch.compile(model)
    return model

def generate_text(model, tokenizer, prompt, device, config_params):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=config_params['max_length'],
            temperature=config_params['temperature'],
            top_p=config_params['top_p'],
            top_k=config_params['top_k'],
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

 