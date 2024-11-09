from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import torch
import mlflow
import logging
from bert_score import score as bert_score

logging.basicConfig(level=logging.DEBUG)

# Initialize MLflow experiment
mlflow.set_experiment("Llama-Text-Generation")

# Load dataset
dataset = load_dataset('allenai/commongen_lite', split='train')

# Check CUDA availability and set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
quantization_config = BitsAndBytesConfig(load_in_4bit=False, load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map={"": device},
    quantization_config=quantization_config,
    torch_dtype=torch.float16
)

# Generate text for the first data point
concept_set = dataset['concept_set'][0]
cleaned_concepts = [concept.split('_')[0] for concept in concept_set]
prompt = f"Generate a meaningful paragraph with at least 750 words using the following concepts: {', '.join(cleaned_concepts)}."

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
tokenizer.pad_token = tokenizer.eos_token
attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)

with mlflow.start_run() as run:
    mlflow.log_param("prompt", prompt)
    mlflow.log_param("model_name", "Llama-2-7b-chat-hf")
    mlflow.log_param("max_length", 750)
    mlflow.log_param("temperature", 0.7)
    mlflow.log_param("top_p", 0.9)
    mlflow.log_param("top_k", 70)

with torch.no_grad():
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=70,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated Sentence: {generated_sentence}")

mlflow.log_text(generated_sentence, "generated_text.txt")

# Example reference sentence
reference = "The dog catches the frisbee when the boy throws it into the air."

# Calculate BERTScore
P, R, F1 = bert_score([generated_sentence], [reference], lang="en")
print(f"BERTScore - Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}")

# Log BERTScore results to MLflow
mlflow.log_metric("BERTScore_Precision", P.mean().item())
mlflow.log_metric("BERTScore_Recall", R.mean().item())
mlflow.log_metric("BERTScore_F1", F1.mean().item())
