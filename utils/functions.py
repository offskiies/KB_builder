import json
import os

import requests
from openai import OpenAI

from prompts import assistant_instructions, spr_prompt

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
AIRTABLE_API_KEY = os.environ['AIRTABLE_API_KEY']

# Init OpenAI Client
client = OpenAI(api_key=OPENAI_API_KEY)







# Create or load assistant
def create_assistant(client):
  assistant_file_path = 'assistant.json'

  # If there is an assistant.json file already, then load that assistant
  if os.path.exists(assistant_file_path):
    with open(assistant_file_path, 'r') as file:
      assistant_data = json.load(file)
      assistant_id = assistant_data['assistant_id']
      print("Loaded existing assistant ID.")
  else:
    # If no assistant.json is present, create a new assistant using the below specifications

    # To change the knowledge document, modifiy the file name below to match your document
    # If you want to add multiple files, paste this function into ChatGPT and ask for it to add support for multiple files
    file = client.files.create(file=open("knowledge.txt", "rb"),
                               purpose='assistants')

    assistant = client.beta.assistants.create(
        instructions=assistant_instructions,
        model="gpt-4-1106-preview",
        tools=[
            {
                "type": "retrieval"  # This adds the knowledge base as a tool
            },
            {
                "type": "function",  # This adds the content summarization as a tool
                "function": {
                    "name": "transcribe_and_summarize",
                    "description":
                    "Transcribe the provided content (e.g. youtube link, article link, long text, etc)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content_link": {
                                "type": "string",
                                "description": "Link to the content for transcription or long text."
                            }
                        },
                        "required": ["content_link"]
                    }
                }
            },
            {
                "type": "function",  # This adds the knowledge base integration as a tool
                "function": {
                    "name": "add_to_knowledge_base",
                    "description":
                    "Integrate the summary into the user's personal knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "The summary to be added to the knowledge base."
                            }
                        },
                        "required": ["summary"]
                    }
                }
            }
        ],
        file_ids=[file.id]
    )

    # Create a new assistant.json file to load on future runs
    with open(assistant_file_path, 'w') as file:
      json.dump({'assistant_id': assistant.id}, file)
      print("Created a new assistant and saved the ID.")

    assistant_id = assistant.id

  return assistant_id


# Add lead to Airtable
def create_lead(name, phone, address):
  url = "https://api.airtable.com/v0/appM1yx0NobvowCAg/Leads"  # Change this to your Airtable API URL
  headers = {
      "Authorization": AIRTABLE_API_KEY,
      "Content-Type": "application/json"
  }
  data = {
      "records": [{
          "fields": {
              "Name": name,
              "Phone": phone,
              "Address": address
          }
      }]
  }
  response = requests.post(url, headers=headers, json=data)
  if response.status_code == 200:
    print("Lead created successfully.")
    return response.json()
  else:
    print(f"Failed to create lead: {response.text}")
