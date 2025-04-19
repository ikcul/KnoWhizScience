# Pipeline Service

## Installation

```bash
conda create --name knowhiz python=3.11
conda activate knowhiz
# pip install langchain openai unstructured pdf2image pdfminer pdfminer.six "langchain[docarray]" tiktoken scipy faiss-cpu pandas pymupdf langchain_openai langchain_community langchain-anthropic scikit-learn
# pip install quart quart-cors celery "celery[redis]" gevent eventlet pymongo azure-core azure-storage-blob
# pip install discord.py
# pip install moviepy pydub wikipedia wikipedia-api youtube-transcript-api
pip install -r requirements.txt
```

install MacTex or TeX Live

```bash
# e.g. on macOS or Linux
brew install --cask mactex
```

install ffmpeg

```bash
# e.g. on macOS or Linux
brew install ffmpeg
```

Once installed, you can set the IMAGEIO_FFMPEG_EXE environment variable as indicated in your script. This variable points to the FFmpeg executable, which is typically located in /usr/local/bin/ffmpeg on macOS, but the provided script suggests a Homebrew-specific path under /opt/homebrew/bin/ffmpeg. Verify the correct path using:

```bash
which ffmpeg
```

Then update the environment variable accordingly in your Python script or set it in your shell profile:

```bash
export IMAGEIO_FFMPEG_EXE=$(which ffmpeg)
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
```

## Set OPENAI_API_KEY

```bash
cd knowhizService
# Should replace sk-xxx to a real openai api key
echo "OPENAI_API_KEY=sk-xxx" > .env
```

## Run Native

```bash
# Copy the pdf file to knowhizService/pipeline/test_inputs/ folder
conda activate knowhiz
cd knowhizService
python local_test.py <filename>
```
