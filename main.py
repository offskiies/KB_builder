from utils.knowledge_graph import check_if_kg_exists, plot_2D_kg, plot_3D_kg
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.embeddings import OpenAIEmbeddings
from langchain import HuggingFaceHub, VectorDBQA
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone
from langchain.chains import VectorDBQA
from matplotlib import pyplot as plt
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
import moviepy.editor as mp
from typing import Optional
from utils import prompts
import gradio as gr
import pinecone
import whisper
import os


plt.switch_backend("Agg")
load_dotenv()

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = "us-east4-gcp"
embed_model = "text-embedding-ada-002"

if not os.path.isdir("data"):
    os.mkdir("data")

# Pinecone Setup
pinecone.init(
        api_key = PINECONE_API_KEY,
        environment = PINECONE_ENV
)
# Set the index name for this project in pinecone first
index_name = 'kb-builder'


llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
chat = ChatOpenAI(model="gpt-4-1106-preview",temperature=0, openai_api_key=OPENAI_API_KEY)


text_splitter = RecursiveCharacterTextSplitter(
    separators=["."], chunk_size=1000, chunk_overlap=25
)

t5 = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    huggingfacehub_api_token=os.environ["HUGGINGFACE_API_KEY"],
    model_kwargs={"max_new_tokens": 500, "temperature": 1e-10},
)


summariser = load_summarize_chain(
    t5,
    chain_type="map_reduce",
    map_prompt=prompts.map_reduce_prompt,
    return_intermediate_steps=True,
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def transcribe(content, content_type, model_size: Optional[str] = "tiny.en"):
    global transcript
    print(f"content type is {content_type}")
    if content_type == "Youtube":
        video_id = content.replace('https://www.youtube.com/watch?v=', '')
        result = YouTubeTranscriptApi.get_transcript(video_id)
        transcript=''
        for x in result:
            sentence = x['text']
            transcript += f' {sentence}\n'

    elif content_type == "Article":
        print("Article")
        transcript="Article is the transcription world of hi there matey"

    elif content_type == "Tweet":
        print("Tweet")
        transcript="Tweet is the transcription world of hi there matey"

    elif content_type == "Video":
        transcriber_model = whisper.load_model(transcriber_model_sizes[model_size])
        # First we must convert the video to Audio
        mp.VideoFileClip(content).audio.write_audiofile("data/Convertedaudio.wav")
        output = transcriber_model.transcribe("data/Convertedaudio.wav")
        transcript = output["text"].strip()

    elif content_type == "Audio":
        transcriber_model = whisper.load_model(transcriber_model_sizes[model_size])
        # Extract the audio data from the tuple
        audio_data = content[1] if isinstance(content, tuple) else content
        mp.AudioFileClip(content).write_audiofile("data/audio.wav")
        output = transcriber_model.transcribe(audio_data)
        transcript = output["text"].strip()

    else:
        return f'Error, content type({content_type}) is not found!'

    return transcript

summaries = []


def add_to_knowledge_base(summary):
    print("Adding summary to vector KB")
    global texts
    texts = text_splitter.create_documents([summary])

    if index_name not in pinecone.list_indexes():
        print("Index does not exist: ", index_name)

    try:
        db = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name = index_name)
        global qa
        qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=db)
        return "Successfully Added to Vector DB!"
    
    except SystemError:
        return "Error: Could not add summary to DB"

def summarise():
    messages = [
        SystemMessage(content=prompts.spr_prompt),
        HumanMessage(content=f"Here is a transcript, please summarise it: {transcript}"),
    ]
    
    summary = chat.invoke(messages).content

    return summary


def question_answer(question: str):
    try:
        return qa.run(question)
       
    except NameError:
        return "Error: Video has not yet been transcribed"


def generate_2D_knowledge_graph(filter: Optional[str] = None):
    check_if_kg_exists([doc.page_content for doc in texts])
    return plot_2D_kg(filter)


def generate_3D_knowledge_graph(filter: Optional[str] = None):
    check_if_kg_exists([doc.page_content for doc in texts])
    return plot_3D_kg(filter)


transcriber_model_sizes = ["tiny.en","base.en","small.en","medium","large-v3"]
summary_lengths = {"Short": "5-10 WORD","Medium": "30-50 WORD","Long": "75-125 WORD"}
content_types = ["Youtube","Article","Tweet"]

qa_examples = [
    ["How is AI being democratised?"],
    ["What is the future of AI?"],
    ["Who is predominantly implementing AI currently?"],
    ["Why aren't small companies benefitting from machine learning?"],
    ["Give me a detailed example of how a small company could benefit from AI?"]]

kg_examples = [["AI"], ["Tech"], ["Project"]]



with gr.Blocks(theme="ParityError/Interstellar") as demo: 
    gr.Label(value="Knowledge Base Builder", show_label=False, color="#560071", container=False)
    with gr.Tab("Links"):
        with gr.Row():
            content_type = gr.Dropdown(
                    choices=content_types,
                    value="Youtube",
                    multiselect=False,
                    label="Link Type:",
                    interactive=True,
                )
            link = gr.Textbox(label="Source Link", placeholder="Please enter the link here.")
    
        transcript_txt = gr.Textbox(label=f"Transcript", placeholder=f"The transcript will appear here.")

        transcribe_btn = gr.Button("Transcribe")
        transcribe_btn.click(fn=transcribe, inputs=[link, content_type], outputs=transcript_txt)


    with gr.Tab("Upload Audio"):
        content_type = gr.Dropdown(
                    choices=["Audio"],
                    value="Audio",
                    multiselect=False,
                    interactive=False,
                    visible=False
                )
        audio_file = gr.Audio(sources=['microphone','upload'], type="filepath")
    
        transcript_txt = gr.Textbox(label="Audio Transcript", placeholder="Audio transcript will appear here.")

        model_size = gr.Slider(minimum=0, maximum=4, step=1,value=1,label="Transcription Model Size:",
                    info="Increasing model size will increase transcription accuracy at the cost of speed.",interactive=True)

        transcribe_btn = gr.Button("Transcribe")
        transcribe_btn.click(fn=transcribe, inputs=[audio_file, content_type, model_size], outputs=transcript_txt)

    with gr.Tab("Upload Video"):
        content_type = gr.Dropdown(
                    choices=["Video"],
                    value="Video",
                    multiselect=False,
                    interactive=False,
                    visible=False
                )
        video_file = gr.Video()
    
        transcript_txt = gr.Textbox(label="Video Transcript", placeholder="Video transcript will appear here.")

        model_size = gr.Slider(minimum=0, maximum=4, step=1,value=1,label="Transcription Model Size:",
                    info="Increasing model size will increase transcription accuracy at the cost of speed.",interactive=True)

        transcribe_btn = gr.Button("Transcribe")
        transcribe_btn.click(fn=transcribe, inputs=[video_file, content_type, model_size], outputs=transcript_txt)


    with gr.Blocks():
        with gr.Tab("Summarise:"):
            # summary_dropdown = gr.Dropdown(
            #     choices=list(summary_lengths.keys()),
            #     value="Short",
            #     multiselect=False,
            #     label="Summary length:",
            #     info="Select the length of summary to be generated.",
            #     interactive=True,
            # )

            summary = gr.TextArea(
                label="Summary:",
                placeholder="The summary will appear here.",
                info="After the summary appears, you can ammend it or add more notes then append it to KB",
                interactive=True,
            )

            append_status = gr.Label(container=False, scale=0.2)


            with gr.Row():
                summary_btn = gr.Button("Summarise")
                summary_btn.click(fn=summarise, outputs=summary)
                append_btn = gr.Button("Append to KB")
                append_btn.click(fn=add_to_knowledge_base, inputs= summary, outputs=append_status)

        with gr.Tab("Chat with KB:"):
            question = gr.Textbox(
                label="Question:",
                placeholder="Type your question on the contents of the video.",
            )
            answer = gr.Textbox(
                label="Answer:",
                placeholder="Answer from the model will appear here.",
                interactive=False,
            )

            

            qa_btn = gr.Button("Submit")
            qa_btn.click(fn=question_answer, inputs=question, outputs=answer)

        # with gr.Tab("Knowledge Graph:"):
        #     with gr.Row():
        #         kg_filter = gr.Textbox(
        #             label="Filter:",
        #             placeholder="Use to filter displayed knowledge graph.",
        #         )

        #         gr.Examples(kg_examples, inputs=kg_filter, label="Example Filters:")

        #     with gr.Row():
        #         kg_2D_button = gr.Button("Generate 2D Knowledge Graph")
        #         kg_3D_button = gr.Button("Generate 3D Knowledge Graph")

        #     knowledge_graph = gr.Plot()

        #     kg_2D_button.click(
        #         fn=generate_2D_knowledge_graph,
        #         inputs=kg_filter,
        #         outputs=knowledge_graph,
        #     )

        #     kg_3D_button.click(
        #         fn=generate_3D_knowledge_graph,
        #         inputs=kg_filter,
        #         outputs=knowledge_graph,
            # )

demo.launch()
