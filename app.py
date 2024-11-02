from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)


# Load the saved model and tokenizer
model_save_path = 'artifacts\model_trainer/saved_model'  # Replace with your actual model path

# Load the model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_save_path,
    load_in_8bit=True,  # Enable 8-bit loading
    torch_dtype=torch.float16,  # Use mixed precision for memory efficiency
)
tokenizer = AutoTokenizer.from_pretrained(model_save_path)

# Check CUDA availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Define device

@app.route('/')
def index():
    return render_template('index.html')  # Render input form

@app.route('/generate', methods=['POST'])
def generate():
    # Get the input from the form
    concept_input = request.form['concepts']
    
    # Process the input
    cleaned_concepts = concept_input.split(",")  # Split input by commas
    prompt = f"Generate a meaningful paragraph with at least 750 words using the following concepts: {', '.join(cleaned_concepts)}."

    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids  # Keep input_ids on CPU

    # Move input_ids to device
    input_ids = input_ids.to(device)  # Move input_ids to device

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=1024,
            temperature=0.7,
            top_p=0.9,
            top_k=70,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

    # Render result in the template
    return render_template('results.html', sentence=generated_sentence)

if __name__ == '__main__':
    app.run(debug=True)


