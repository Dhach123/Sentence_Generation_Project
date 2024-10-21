from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import mlflow

# Initialize MLflow experiment
mlflow.set_experiment("Llama-Text-Generation")

# Load dataset
dataset = load_dataset('allenai/commongen_lite', split='train')

# Check CUDA availability and set device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Load the model with CPU offloading and 8-bit precision
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",  # Change to local path if necessary
    device_map={"": device},  # Manually set device allocation
    load_in_8bit=True,  # Enable 8-bit loading for reduced memory usage
    torch_dtype=torch.float16  # Use mixed precision for memory efficiency
)

# Example: Generate text for the first data point
concept_set = dataset['concept_set'][0]  # Example concept set
# Remove tags from concepts (keeping only the base word)
cleaned_concepts = [concept.split('_')[0] for concept in concept_set]

# Create prompt
prompt = f"Generate a meaningful paragraph with at least 750 words using the following concepts: {', '.join(cleaned_concepts)}."

# Tokenize the prompt
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)  # Move input_ids to the device

# Set pad_token to eos_token to avoid errors
tokenizer.pad_token = tokenizer.eos_token

# Track experiment parameters with MLflow
with mlflow.start_run() as run:
    # Log the prompt and other parameters
    mlflow.log_param("prompt", prompt)
    mlflow.log_param("model_name", "Llama-2-7b-chat-hf")
    mlflow.log_param("max_length", 1024)
    mlflow.log_param("temperature", 0.7)
    mlflow.log_param("top_p", 0.9)
    mlflow.log_param("top_k", 70)

# Generate text
with torch.no_grad():  # Disable gradient calculation to save memory
    output = model.generate(
        input_ids,
        max_length=1024,  # Adjusted max_length to allow for more content
        temperature=0.7,
        top_p=0.9,
        top_k=70,
        num_return_sequences=1,  # Generate only one sequence to save memory
        pad_token_id=tokenizer.eos_token_id  # To avoid an error for missing pad_token_id
    )

# Decode and print the generated text
generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated Sentence: {generated_sentence}")



# Log the generated text
mlflow.log_text(generated_sentence, "generated_text.txt")
    
# Optionally, log the model
mlflow.pytorch.log_model(model, "model")
