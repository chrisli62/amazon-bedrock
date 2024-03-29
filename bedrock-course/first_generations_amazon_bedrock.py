#Import all packages

#block
import boto3
import json

#Setup Bedrock runtime
#block
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')

#block
prompt = "Write a one sentence summary of Las Vegas."

#block
kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt
        }
    )
}

#block
response = bedrock_runtime.invoke_model(**kwargs)

#block
response

#block
response_body = json.loads(response.get('body').read())

#block
print(json.dumps(response_body, indent=4))

#block
print(response_body['results'][0]['outputText'])

#Generation Configuration

#block
prompt = "Write a summary of Las Vegas."

#block
kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}

#block
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)

#block
print(json.dumps(response_body, indent=4))

#block
kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 500,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}

#block
response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)

#block
print(json.dumps(response_body, indent=4))

#Working with other type of data

#block
from IPython.display import Audio

#block
audio = Audio(filename="dialog.mp3")
display(audio)

#block
with open('transcript.txt', "r") as file:
    dialogue_text = file.read()

#block
print(dialogue_text)

#block
prompt = f"""The text between the <transcript> XML tags is a transcript of a conversation. 
Write a short summary of the conversation.

<transcript>
{dialogue_text}
</transcript>

Here is a summary of the conversation in the transcript:"""

#block
print(prompt)

#block
kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0,
                "topP": 0.9
            }
        }
    )
}

#block
response = bedrock_runtime.invoke_model(**kwargs)

#block
response_body = json.loads(response.get('body').read())
generation = response_body['results'][0]['outputText']

#block
print(generation)
