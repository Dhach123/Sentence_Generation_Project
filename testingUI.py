from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Set pad_token to eos_token to avoid errors later
tokenizer.pad_token = tokenizer.eos_token

# Load the model with CPU offloading and 8-bit precision (no need to move it to device manually)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",  
    load_in_8bit=True,  # Enable 8-bit loading for reduced memory usage
    torch_dtype=torch.float16
    # device_map and model is already loaded on the appropriate device, no need for .to(device)
)

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
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)  # Keep device assignment here
    
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