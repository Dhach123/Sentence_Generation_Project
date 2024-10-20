from textgeneration.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from textgeneration.pipeline.stage02_data_validation import DataValidationTrainingPipeline
from textgeneration.pipeline.stage03_data_transformation import DataTransformationTrainingPipeline
from textgeneration.pipeline.stage04_model_trainer import DataModelTrainingPipeline
from textgeneration.logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e



STAGE_NAME = "Data transformaation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from textgeneration.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textgeneration.utils.common import read_yaml, create_directories

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str

# Instantiate the configuration
config = ModelTrainerConfig(
    root_dir=Path('artifacts/model_trainer'),
    data_path=Path('artifacts/data_transformation/cleaned_concepts.csv'),
    model_ckpt='meta-llama/Llama-2-7b-chat-hf',  # Correct model ID
)

class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_ckpt=config.model_ckpt,
        )

        return model_trainer_config

def main():
    STAGE_NAME = "Pretrained Model Stage"

    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        
        # Initialize the ConfigurationManager
        config_manager = ConfigurationManager()
        
        # Get the model trainer configuration
        model_trainer_config = config_manager.get_model_trainer_config()

        # Load the dataset from the configuration
        dataset = pd.read_csv(model_trainer_config.data_path)

        # Check CUDA availability and set device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_trainer_config.model_ckpt)

        # Load the model with CPU offloading and 8-bit precision
        model = AutoModelForCausalLM.from_pretrained(
            model_trainer_config.model_ckpt,
            device_map={"": device},
            load_in_8bit=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # Optionally, compile the model for faster execution (requires PyTorch 2.0+)
        if torch.__version__ >= "2.0":
            model = torch.compile(model)

        # Example: Generate text for the first data point
        concept_set = dataset['cleaned_concept_set'][100]  # Example concept set
        
        # Create prompt
        prompt = f"Generate a meaningful paragraph with at least 750 words using the following concepts: {''.join(concept_set)}."

        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)  # Move input_ids to the device

        # Set pad_token to eos_token to avoid errors
        tokenizer.pad_token = tokenizer.eos_token

        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=config_manager.params['TrainingArguments']['max_length'],
                temperature=config_manager.params['TrainingArguments']['temperature'],
                top_p=config_manager.params['TrainingArguments']['top_p'],
                top_k=config_manager.params['TrainingArguments']['top_k'],
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode and print the generated text
        generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated Sentence: {generated_sentence}")

        # Save the model and tokenizer
        model_save_path = os.path.join(model_trainer_config.root_dir, "saved_model")
        os.makedirs(model_save_path, exist_ok=True)

        # Save the model and tokenizer
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        print(f"Model and tokenizer saved to {model_save_path}")
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        # Handle any exceptions and print the error message
        logger.error(f"An error occurred during stage {STAGE_NAME}: {e}")
        raise e

if __name__ == "__main__":
    main()
