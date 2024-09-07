# app.py
from flask import Flask, render_template, request, Response, stream_with_context, jsonify
import requests
import json
from requests.exceptions import ConnectionError, RequestException, Timeout
import logging
import threading

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class StoppableGenerator:
    def __init__(self, generator):
        self.generator = generator
        self.stop_event = threading.Event()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_event.is_set():
            raise StopIteration
        return next(self.generator)

    def stop(self):
        self.stop_event.set()

class LMStudioProxy:
    def __init__(self, base_url="http://192.168.1.21:9001"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.models = [
            "bartowski/stable-code-instruct-3b-GGUF/stable-code-instruct-3b-Q4_0.gguf",
            "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "second-state/Llava-v1.5-7B-GGUF/llava-v1.5-7b-Q4_0.gguf",
            "internlm/internlm2_5-20b-chat-gguf/internlm2_5-20b-chat-q4_0.gguf",
            "lmstudio-community/Codestral-22B-v0.1-GGUF/Codestral-22B-v0.1-Q4_K_M.gguf",
            "TheBloke/WizardCoder-Python-34B-V1.0-GGUF/wizardcoder-python-34b-v1.0.Q3_K_S.gguf",
            "TheBloke/WizardCoder-33B-V1.1-GGUF/wizardcoder-33b-v1.1.Q3_K_S.gguf"
        ]

    def get_models(self):
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            models = response.json()
            all_models = list(set(self.models + [model['id'] for model in models['data']]))
            return {'data': [{'id': model} for model in all_models]}
        except (ConnectionError, RequestException, Timeout) as e:
            logging.error(f"Error fetching models: {str(e)}")
            return {'data': [{'id': model} for model in self.models]}

    def generate(self, model, prompt, stream=True):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": -1,
            "stream": stream
        }

        try:
            response = requests.post(f"{self.base_url}/v1/chat/completions", 
                                    headers=self.headers, 
                                    json=payload, 
                                    stream=stream,
                                    timeout=30)
            response.raise_for_status()
            
            if stream:
                return self._process_stream(response)
            else:
                return response.json()
        except ConnectionError as e:
            logging.error(f"Connection error: {str(e)}")
            raise Exception("Unable to connect to the LM Studio server. Please check if it's running and accessible.")
        except Timeout as e:
            logging.error(f"Timeout error: {str(e)}")
            raise Exception("The request to LM Studio server timed out. Please try again later.")
        except RequestException as e:
            logging.error(f"Request error: {str(e)}")
            raise Exception(f"Error communicating with the LM Studio server: {str(e)}")

    def _process_stream(self, response):
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8').strip()
                logging.debug(f"Received line: {line}")
                if line.startswith('data: '):
                    try:
                        data = json.loads(line[6:])
                        if 'choices' in data and len(data['choices']) > 0:
                            content = data['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding JSON: {str(e)}, Line: {line}")
                        yield f"Error: Invalid response format from server"
                elif line == 'data: [DONE]':
                    logging.info("Stream completed")
                    break
                else:
                    logging.warning(f"Unexpected line format: {line}")
                    yield f"Warning: Unexpected response format from server"

lm_proxy = LMStudioProxy()
generators = {}
app = Flask(__name__)
session = {}
uuid = __import__('uuid')

@app.route('/generate', methods=['POST'])
def generate():
    model = request.form.get('model', '')
    prompt = request.form.get('prompt', '')
    session_id = request.form.get('session_id', '')

    logging.info(f"Generating with model: {model}, prompt: {prompt}, session_id: {session_id}")

    def generate_stream():
        full_response = ""
        try:
            stoppable_gen = StoppableGenerator(lm_proxy.generate(model, prompt))
            generators[session_id] = stoppable_gen
            for content in stoppable_gen:
                full_response += content
                yield f"data: {json.dumps({'content': content})}\n\n"
            yield "data: [DONE]\n\n"
            
            # Add the conversation to the history
            session['conversation_history'].append({
                'prompt': prompt,
                'response': full_response
            })
            session.modified = True
        except Exception as e:
            logging.error(f"Error during generation: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if session_id in generators:
                del generators[session_id]

    return Response(stream_with_context(generate_stream()), content_type='text/event-stream')


@app.route('/stop', methods=['POST'])
def stop_generation():
    session_id = request.form.get('session_id', '')
    if session_id in generators:
        generators[session_id].stop()
        del generators[session_id]
        return jsonify({"status": "stopped"}), 200
    return jsonify({"status": "not found"}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {str(e)}")
    return jsonify(error=str(e)), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    session['conversation_history'] = []
    session.modified = True
    return jsonify({"status": "cleared"}), 200

@app.route('/')
def index():
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    models = lm_proxy.get_models()
    return render_template('index.html', models=models['data'], conversation_history=session['conversation_history'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)