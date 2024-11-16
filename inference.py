from flask import Flask, request, render_template, send_file, jsonify
import os
from werkzeug.utils import secure_filename
import yaml
import time
from ml_collections import ConfigDict
import torch
import soundfile as sf
from pydub import AudioSegment
import threading
from utils import demix_track, get_model_from_config
import torch.nn as nn
import requests
from tqdm import tqdm


app = Flask(__name__)

# Глобальные настройки
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
MODEL_FOLDER = 'models'
CONFIG_FOLDER = 'configs'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}
processing_lock = threading.Lock()
current_task = None

# URLs для скачивания
MODEL_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"
CONFIG_URL = "https://raw.githubusercontent.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/main/configs/config_vocals_mel_band_roformer.yaml"


# Создаем необходимые папки
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_FOLDER, CONFIG_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def download_from_url(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {destination}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=destination,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

def initialize_files():
    model_path = os.path.join(MODEL_FOLDER, 'MelBandRoformer.ckpt')
    config_path = os.path.join(CONFIG_FOLDER, 'config_vocals_mel_band_roformer.yaml')
    
    # Update the function call to use the new name
    download_from_url(MODEL_URL, model_path)
    download_from_url(CONFIG_URL, config_path)
    
    return model_path, config_path

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_wav(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format='wav')

class ProcessingTask:
    def __init__(self, filename):
        self.filename = filename
        self.status = "processing"
        self.progress = 0
        self.output_files = []

def process_audio(model, config, device, input_path, output_dir, task):
    try:
        with processing_lock:
            model.eval()
            
            # Конвертируем в WAV если нужно
            file_ext = os.path.splitext(input_path)[1].lower()
            if file_ext != '.wav':
                wav_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + '.wav')
                convert_to_wav(input_path, wav_path)
                input_path = wav_path

            mix, sr = sf.read(input_path)
            mixture = torch.tensor(mix.T, dtype=torch.float32)

            instruments = config.training.instruments
            if config.training.target_instrument is not None:
                instruments = [config.training.target_instrument]

            res, _ = demix_track(config, model, mixture, device)

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results and store only filenames
            for instr in instruments:
                output_filename = f"{base_name}_{instr}.wav"
                output_path = os.path.join(output_dir, output_filename)
                sf.write(output_path, res[instr].T, sr, subtype='FLOAT')
                task.output_files.append(output_filename)  # Store only filename

            # Save instrumental
            vocals = res[instruments[0]].T
            instrumental = mix - vocals
            instrumental_filename = f"{base_name}_instrumental.wav"
            instrumental_path = os.path.join(output_dir, instrumental_filename)
            sf.write(instrumental_path, instrumental, sr, subtype='FLOAT')
            task.output_files.append(instrumental_filename)  # Store only filename

            task.status = "completed"
    except Exception as e:
        task.status = "error"
        task.error_message = str(e)
        print(f"Error processing audio: {str(e)}")


def initialize_model(config_path, model_path):
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    model = get_model_from_config('mel_band_roformer', config)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
    else:
        device = 'cpu'
        model = model.to(device)

    return model, config, device

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_task
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        current_task = ProcessingTask(filename)
        
        # Запускаем обработку в отдельном потоке
        thread = threading.Thread(
            target=process_audio,
            args=(model, config, device, input_path, OUTPUT_FOLDER, current_task)
        )
        thread.start()

        return jsonify({
            'message': 'Processing started',
            'filename': filename
        })

    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/status')
def get_status():
    if current_task is None:
        return jsonify({'status': 'no_task'})
    
    return jsonify({
        'status': current_task.status,
        'filename': current_task.filename,
        'progress': current_task.progress,
        'output_files': current_task.output_files if current_task.status == 'completed' else []
    })

@app.route('/download/<filename>')
def download_file(filename):
    # Ensure the filename is secure
    filename = secure_filename(filename)
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
        
    try:
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename  # Explicitly set download name
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    model_path, config_path = initialize_files()
    model, config, device = initialize_model(config_path, model_path)
    app.run(host='0.0.0.0', port=5000)