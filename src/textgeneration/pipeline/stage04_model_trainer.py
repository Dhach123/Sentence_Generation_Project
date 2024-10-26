from textgeneration.config.configuration import ConfigurationManager
from textgeneration.conponents.Model_trainer import ModelTrainerConfig
from textgeneration.logging import logger
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class DataModelTrainingPipeline:
    def __init__(self, config, params):
        self.config = config
        self.params = params

    def main(self):
        # Load the dataset
        dataset = pd.read_csv(self.config['model_trainer']['data_path'])

        # Check CUDA availability and set device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_trainer']['model_ckpt'])

        # Load the model with CPU offloading and 8-bit precision
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model_trainer']['model_ckpt'],
            device_map={"": device},
            load_in_8bit=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Compile the model for faster execution (requires PyTorch 2.0+)
        if torch.__version__ >= "2.0":
            model = torch.compile(model)

        # Example: Generate text for the first data point
        concept_set = dataset['cleaned_concept_set'][0]

        # Create the prompt
        prompt = f"Generate a meaningful paragraph with at least 750 words using the following concepts: {''.join(concept_set)}."

        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

        # Set pad_token to eos_token to avoid errors
        tokenizer.pad_token = tokenizer.eos_token

        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=self.params['TrainingArguments']['max_length'],
                temperature=self.params['TrainingArguments']['temperature'],
                top_p=self.params['TrainingArguments']['top_p'],
                top_k=self.params['TrainingArguments']['top_k'],
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and print the generated text
        generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated Sentence: {generated_sentence}")

        # Save the model and tokenizer
        model_save_path = os.path.join(self.config['model_trainer']['root_dir'], "saved_model")
        os.makedirs(model_save_path, exist_ok=True)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        print(f"Model and tokenizer saved to {model_save_path}")
