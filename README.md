# Knowledge Base Builder
Build your own custom knowledge base from various sources such as youtube videos transcripts, tweets, articles, videos and audios. Uses Gradio for the UI

## Initial set-up:
1. Create `.env` file in root of directory to store Pinecone and openai API keys.
```
OPENAI_API_KEY='<YOUR OPENAI API KEY>'
PINECONE_API_KEY='<YOUR PINECONE_API_KEY>'
```
2. Install dependencies.
```shell
brew install ffmpeg
```
3. Install environment.
```shell
pyenv install 3.10.4
pyenv shell 3.10.4
poetry env use 3.10.4
poetry install
```

## Run the demo:

1. Run main.py
```shell
poetry run python main.py
```
2. Open: http://127.0.0.1:7860

## Linting:
```shell
poetry run black -l 80 .
```
