from utils.functions import *
import gradio as gr
import os


if not os.path.isdir("data"):
    os.mkdir("data")


content_types = ["Youtube","Article","Tweet"]


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


demo.launch()
