import os
import shutil
import subprocess

import warnings

import yt_dlp

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # pydub warns if ffmpeg isn't on PATH
    from pydub import AudioSegment

DOWNLOAD_DIR = 'downloades'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_ffmpeg_path() -> str:
    """Return an ffmpeg executable: system ffmpeg if on PATH, else the imageio-ffmpeg bundled binary."""
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg and add it to PATH, "
            "or `pip install imageio-ffmpeg`."
        ) from e


# Let pydub use the same ffmpeg binary (system or bundled).
try:
    AudioSegment.converter = get_ffmpeg_path()
except RuntimeError:
    pass


def download_youtube_audio(url: str) -> str:
    """Download the best audio stream and return the path of the downloaded file."""
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloads = info.get("requested_downloads") or []
        if downloads and downloads[0].get("filepath"):
            return downloads[0]["filepath"]
        return ydl.prepare_filename(info)


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono 16-bit WAV using ffmpeg."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    cmd = [
        get_ffmpeg_path(), "-y", "-loglevel", "error",
        "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.strip()}")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    source = source.strip().strip('"').strip("'")
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        downloaded = download_youtube_audio(source)
        print("Converting to WAV...")
        wav_path = convert_to_wav(downloaded)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
