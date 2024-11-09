# Sentence_Generation_Project

Workflows
Update config.yaml
Update params.yaml
Update entity
Update the configuration manager in src config
update the components
update the pipeline
update the main.py
update the app.py

How to run?

STEPS:

Clone repository https://github.com/Dhach123/Sentence_Generation_Project.git

STEP 01- Create a conda environment after opening the repository
 source ~/anaconda3/etc/profile.d/conda.sh
 conda activate myenv
code . # launch VS CODE


STEP 02- install the requirements
python -m pip install -r requirements.txt


# Finally run the following command
python app.py

open up you local host and port for UI Interface 

# Experiment Tracking for hyperparametertuning 
MLFLOW for experiment tracking is used

# Deployment 
Model is deployed locally due to resource constraints for this heavy model

# KEY Highlites and Important Notes

1) Make sure you create .ignore file in project setup to untrack Large artifacts project file of models/meta-llama/Llama-2-7b-chat-hf/ 
2) To avoid CUDA out of memory error the LLAMA model was loaded using bitandbites library
3) My GPU memory is 4GB
4) Handled lot of challenges to fix CUDA out of memory 
5) Pretrained LLAMA model used and avoided model training. 
6) Model results were as expected

