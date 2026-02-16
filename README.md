# Kazakh Video Dubbing

An automated video dubbing system that translates and dubs videos from English to Kazakh with synchronized speech and subtitle support.

## Features

- **Accurate Speech Recognition**: Uses OpenAI Whisper for precise transcription
- **Dual Translation Methods**: Supports both Google Translate and GPT-4 for translation
- **Speaker Gender Detection**: Automatically selects appropriate voice based on speaker gender
- **Speech Synchronization**: Syncs translated speech with video timing
- **High-Quality TTS**: Supports multiple text-to-speech engines (Edge TTS, Yandex SpeechKit)
- **Subtitle Support**: Generates synchronized subtitles in SRT format
- **Web Interface**: User-friendly Flask-based web application
- **Batch Processing**: Command-line interface for processing multiple videos

## Installation

### Prerequisites

- Python 3.10 or higher
- ffmpeg installed on your system

### Install ffmpeg

```bash
# Windows (using Chocolatey)
choco install ffmpeg

# Linux
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Install Python Dependencies

1. Clone the repository:
```bash
git clone https://github.com/stukenov/kazakh-video-dubbing.git
cd kazakh-video-dubbing
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
- `OPENAI_API_KEY`: For GPT-4 translation (optional)
- `YANDEX_SPEECHKIT_API_KEY`: For Yandex TTS (optional)
- Other API keys as needed

## Usage

### Command Line Interface

#### Basic Usage (Google Translate + Edge TTS)
```bash
python dubbing.py --input input.mp4 --output output.mp4
```

#### Using GPT-4 for Translation
```bash
python dubbing.py --input input.mp4 --output output.mp4 --translate chatgpt
```

#### Selecting Whisper Model Size
```bash
# Faster, less accurate
python dubbing.py --input input.mp4 --output output.mp4 --whisper_model medium

# Slower, more accurate
python dubbing.py --input input.mp4 --output output.mp4 --whisper_model large
```

#### Using Yandex TTS
```bash
python dubbing.py --input input.mp4 --output output.mp4 --tts yandex
```

### Web Interface

Start the Flask web application:
```bash
python web/app.py
```

Then open your browser and navigate to `http://localhost:5000`

## Command Line Arguments

- `--input`: Path to input video file (required)
- `--output`: Path for output dubbed video (required)
- `--translate`: Translation method (`google` or `chatgpt`, default: `google`)
- `--whisper_model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`, default: `medium`)
- `--tts`: TTS service (`edge` or `yandex`, default: `edge`)
- `--mode`: Processing mode (`synchronized` or `simple`, default: `synchronized`)

## Translation Methods Comparison

### Google Translate
- ✅ Fast translation
- ✅ No API keys required
- ✅ Stable operation
- ❌ May be less accurate for complex phrases
- ❌ Sometimes loses context

### GPT-4
- ✅ High-quality translation
- ✅ Preserves context and style
- ✅ Better understanding of complex phrases
- ❌ Requires OpenAI API key
- ❌ May be slower

## Recommendations

### For Short Videos (< 5 minutes)
- Use `--whisper_model medium` for quick results
- Google Translate works well for simple dialogues

### For Important Videos
- Use `--whisper_model large` for accurate transcription
- GPT-4 provides the best translation quality

### For Medium Videos (5-15 minutes)
- `--whisper_model base` offers good balance
- Choose translation method based on content complexity

## Project Structure

```
kazakh-video-dubbing/
├── core/                 # Core processing modules
│   ├── transcriber.py   # Speech recognition
│   ├── translator.py    # Translation engine
│   ├── tts.py          # Text-to-speech
│   ├── video_processor.py
│   └── ...
├── services/            # External service integrations
├── web/                 # Flask web application
├── models/              # Data models
├── utils/               # Utility functions
├── dubbing.py          # Main CLI script
└── requirements.txt    # Python dependencies
```

## Troubleshooting

### FFmpeg Error
```bash
# Windows
choco install ffmpeg
# Add ffmpeg to PATH

# Linux/macOS
# Reinstall ffmpeg using package manager
```

### GPT-4 Unavailable
- System automatically falls back to Google Translate
- Check internet connection
- Verify OPENAI_API_KEY in .env file

### Memory Issues
- Use a smaller Whisper model
- Process video in segments
- Close other applications

## License

MIT License - see [LICENSE](LICENSE) file for details

Copyright (c) 2025 Saken Tukenov

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Edge TTS](https://github.com/rany2/edge-tts) for text-to-speech
- FFmpeg for video processing
