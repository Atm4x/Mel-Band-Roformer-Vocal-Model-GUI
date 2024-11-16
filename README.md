# Audio Separator Web Interface

This is a web interface for the Mel-Band-Roformer Vocal Model that separates vocals from music tracks.

## Features
- Simple web interface
- Supports multiple audio formats (MP3, WAV, OGG, FLAC, M4A)
- Automatic model and configuration download
- Real-time processing status updates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Atm4x/Mel-Band-Roformer-Vocal-Model-GUI
cd audio-separator
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Running the Application

Simply run:
```bash
python inference.py
```

The application will:
1. Automatically download the required model and configuration files
2. Start a web server at http://localhost:5000

## Usage

1. Open your web browser and go to http://localhost:5000
2. Upload an audio file using the web interface
3. Wait for processing to complete
4. Download the separated vocal and instrumental tracks

## Notes

- The first run will download the model (~900MB) and configuration files
- Processing time depends on your computer's specifications and the length of the audio file
- Generated files are stored in the `outputs` folder

## Requirements

- Python 3.7 or higher
- At least 6GB RAM
- GPU is recommended but not required

## License

This project is licensed under the MIT License - see the LICENSE file for details.