from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list    


from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str  # Keep as string for model ID    
# Instantiate the configuration
config = DataTransformationConfig(
    root_dir=Path('artifacts/data_transformation'),
    data_path=Path('artifacts/data_ingestion/commongen_lite_train.csv'),
    tokenizer_name='meta-llama/Llama-2-7b-chat-hf'  # Correct model ID
)    
