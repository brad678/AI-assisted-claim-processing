import google.auth
from googleapiclient.discovery import build

def trigger_dataflow(event, context):
    # Authenticate and initialize the Dataflow client
    credentials, project = google.auth.default()
    dataflow = build('dataflow', 'v1b3', credentials=credentials)

    # Define the parameters for the Dataflow job
    job = {
        'jobName': 'process-claims',
        'environment': {
            'tempLocation': 'gs://your-temp-bucket/temp/',
            'zone': 'us-central1-f'
        },
        'parameters': {
            'inputFile': 'gs://your-bucket-name/{}'.format(event['name']),
            'outputTable': 'your_project_id:your_dataset.your_table'
        },
        'containerSpecGcsPath': 'gs://your-bucket-name/templates/your-template.json'
    }

    # Trigger the Dataflow job
    request = dataflow.projects().locations().templates().launch(
        projectId=project,
        location='us-central1',
        body=job
    )
    response = request.execute()
    print('Dataflow job triggered:', response)
