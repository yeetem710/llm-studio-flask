# app.py
from flask import Flask, render_template, request, Response, stream_with_context
from lmstudio_wrapper import LMStudioWrapper
import json

app = Flask(__name__)

MODELS = [
    "bartowski/stable-code-instruct-3b-GGUF/stable-code-instruct-3b-Q4_0.gguf",
    "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "second-state/Llava-v1.5-7B-GGUF/llava-v1.5-7b-Q4_0.gguf",
    "internlm/internlm2_5-20b-chat-gguf/internlm2_5-20b-chat-q4_0.gguf",
    "lmstudio-community/Codestral-22B-v0.1-GGUF/Codestral-22B-v0.1-Q4_K_M.gguf",
    "TheBloke/WizardCoder-Python-34B-V1.0-GGUF/wizardcoder-python-34b-v1.0.Q3_K_S.gguf",
    "TheBloke/WizardCoder-33B-V1.1-GGUF/wizardcoder-33b-v1.1.Q3_K_S.gguf"
]

lmstudio = LMStudioWrapper()

@app.route('/')
def index():
    return render_template('index.html', models=MODELS)

@app.route('/generate', methods=['POST'])
def generate():
    model = request.form['model']
    prompt = request.form['prompt']

    def generate_stream():
        try:
            response = lmstudio.chat_completion(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            for chunk in response:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate_stream()), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
