
import os
from textgeneration.logging import logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset,load_from_disk
import torch
import pandas as pd
from textgeneration.entity import DataTransformationConfig
from pathlib import Path


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    # Define a function to remove tags from the concept set
    def clean_concept_set(self, df):
        # Assuming concept_set is a string representation of a list
        def clean_single_concept_set(concept_set):
            # Check if the concept_set is a string, otherwise handle it as a list
            if isinstance(concept_set, str):
                concepts = eval(concept_set)  # Convert string representation of list to actual list
            else:
                concepts = concept_set  # Use it directly if already a list
            
            # Extract the base concept (remove _N, _V, etc.)
            cleaned_concepts = [concept.split('_')[0] for concept in concepts]  # Remove tags
            return ', '.join(cleaned_concepts)  # Join cleaned concepts into a string

        # Apply the cleaning function to the 'concept_set' column
        df['cleaned_concept_set'] = df['concept_set'].apply(clean_single_concept_set)

        # Keep only the cleaned concepts for training
        formatted_data = df[['cleaned_concept_set']]

        # Save the processed data to a new CSV file
        output_path = Path('artifacts/data_transformation/cleaned_concepts.csv')
        formatted_data.to_csv(output_path, index=False)

        print("Cleaned data saved to:", output_path)