from textgeneration.config.configuration import ConfigurationManager
from textgeneration.conponents.data_transformation import DataTransformation
from textgeneration.logging import logger
import pandas as pd


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
         config = ConfigurationManager()
         data_transformation_config = config.get_data_transformation_config()
         data_transformation = DataTransformation(config=data_transformation_config)
          # Load the dataset again
         df = pd.read_csv(data_transformation_config.data_path)
         data_transformation.clean_concept_set(df)





