#Import needed packages

#block
import boto3
import json
import os

bedrock = boto3.client('bedrock', region_name="us-west-2")

#block
from helpers.CloudWatchHelper import CloudWatch_Helper
cloudwatch = CloudWatch_Helper()

#block
log_group_name = '/my/amazon/bedrock/logs'

#block
cloudwatch.create_log_group(log_group_name)

#block
loggingConfig = {
    'cloudWatchConfig': {
        'logGroupName': log_group_name,
        'roleArn': os.environ['LOGGINGROLEARN'],
        'largeDataDeliveryS3Config': {
            'bucketName': os.environ['LOGGINGBUCKETNAME'],
            'keyPrefix': 'amazon_bedrock_large_data_delivery',
        }
    },
    's3Config': {
        'bucketName': os.environ['LOGGINGBUCKETNAME'],
        'keyPrefix': 'amazon_bedrock_logs',
    },
    'textDataDeliveryEnabled': True,
}

#block
bedrock.put_model_invocation_logging_configuration(loggingConfig=loggingConfig)

#block
bedrock.get_model_invocation_logging_configuration()
"""

"""

#block
bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-west-2")

#block
prompt = "Write an article about the fictional planet Foobar."

kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}

response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']

print(generation)

#block
cloudwatch.print_recent_logs(log_group_name)

#Review logs in AWS console

#block
from IPython.display import HTML
aws_url = os.environ['AWS_CONSOLE_URL']

#block
HTML(f'<a href="{aws_url}" target="_blank">GO TO AWS CONSOLE</a>')
