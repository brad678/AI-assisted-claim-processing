
# This Cloud Function is triggered by a Cloud Storage event (e.g., file upload) and starts a Dataflow job using the template gs://my-dataflow-templates/claims_processing

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
            'tempLocation': 'gs://claims_data_bucket/temp/',
            'zone': 'us-central1-f'
        },
        'parameters': {
            'inputFile': 'gs://claims_data_bucket/new_claim_data.csv',
            'outputTable': 'my-gcp-project:claims_output_data_bucket.processed_claims'
        },
        'containerSpecGcsPath': 'gs://my_dataflow_templates/claims_processing'
    }

    # Trigger the Dataflow job
    request = dataflow.projects().locations().templates().launch(
        projectId=project,
        location='us-central1',
        body=job
    )
    response = request.execute()
    print('Dataflow job triggered:', response)
