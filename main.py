import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, render_template

app = Flask(__name__)

model_path = "whiterabbitneo/WhiteRabbitNeo-33B-v-1"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="cpu",  # Изменено на CPU
    load_in_4bit=False,
    load_in_8bit=True,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def generate_text(instruction):
    tokens = tokenizer.encode(instruction)
    tokens = torch.LongTensor(tokens).unsqueeze(0)
    tokens = tokens.to("cpu")  # Изменено на CPU

    instance = {
        "input_ids": tokens,
        "top_p": 1.0,
        "temperature": 0.5,
        "generate_len": 1024,
        "top_k": 50,
    }

    length = len(tokens[0])
    with torch.no_grad():
        rest = model.generate(
            input_ids=tokens,
            max_length=length + instance["generate_len"],
            use_cache=True,
            do_sample=True,
            top_p=instance["top_p"],
            temperature=instance["temperature"],
            top_k=instance["top_k"],
            num_return_sequences=1,
        )
    output = rest[0][length:]
    string = tokenizer.decode(output, skip_special_tokens=True)
    answer = string.split("USER:")[0].strip()
    return f"{answer}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        prompt = request.form['prompt']
        
        # Объединение промта и входного текста
        combined_input = prompt + " " + input_text
        
        # Генерация продолжения текста
        answer = generate_text(combined_input)
        
        return render_template('index.html', input_text=input_text, prompt=prompt, answer=answer)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

