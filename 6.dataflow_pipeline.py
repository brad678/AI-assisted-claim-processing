# This Cloud Function is triggered by a Cloud Storage event (e.g., file upload) and starts a Dataflow job using the template created above.

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from google.cloud import storage, bigquery
from transformers import BertTokenizer, BertModel
import torch
from googleapiclient import discovery
import json

class ProcessClaims(beam.DoFn):
    def __init__(self, bert_model_name):
        super(ProcessClaims, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.model = BertModel.from_pretrained(bert_model_name)

    def process(self, element):
        # Extract structured data
        structured_data = {
            'ClaimID': element['ClaimID'],
            'VehicleID': element['VehicleID'],
            'ClaimDate': element['ClaimDate'],
            'CasualCode': element['CasualCode']
        }

        # Generate text embeddings
        text_fields = ['CustomerComplaint', 'TechnicianDiagnosis', 'RepairRecommendation']
        embeddings = []
        for field in text_fields:
            inputs = self.tokenizer(element[field], return_tensors='pt', truncation=True, padding=True, max_length=128)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
            embeddings.append(embedding)

        # Concatenate structured data and embeddings
        combined_features = list(structured_data.values()) + embeddings.flatten().tolist()
        return [combined_features]


class PredictCostAndStatus(beam.DoFn):
    def __init__(self, project, model_name, version_name):
        super(PredictCostAndStatus, self).__init__()
        self.project = project
        self.model_name = model_name
        self.version_name = version_name
        self.prediction_client = discovery.build('ml', 'v1')

    def process(self, element):
        # Prepare the input for prediction
        input_data = {'instances': [element]}
        parent = f'projects/my-gcp-project/models/claims_prediction_model/versions/v1'

        # Call AI Platform Prediction
        request = self.prediction_client.projects().predict(name=parent, body=input_data)
        response = request.execute()

        # Extract predictions
        part_costs = response['predictions'][0]['part_costs']
        labor_hours = response['predictions'][0]['labor_hours']
        claim_status = response['predictions'][0]['claim_status']

        # Append predictions to the element
        element['PartCosts'] = part_costs
        element['LaborHours'] = labor_hours
        element['ClaimStatus'] = claim_status
        return [element]

def run(argv=None):
    pipeline_options = PipelineOptions(argv)
    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | 'ReadClaimData' >> beam.io.ReadFromText('gs://claims_data_bucket/new_claim_data.csv', skip_header_lines=1)
            | 'ParseCSV' >> beam.Map(lambda line: dict(zip(['ClaimID', 'VehicleID', 'ClaimDate', 'CasualCode', 'CustomerComplaint', 'TechnicianDiagnosis', 'RepairRecommendation'], line.split(','))))
            | 'ProcessClaims' >> beam.ParDo(ProcessClaims(bert_model_name='bert-base-uncased'))
            | 'PredictCostAndStatus' >> beam.ParDo(PredictCostAndStatus(project='my-gcp-project', model_name='claims_prediction_model', version_name='v1'))
            | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
                'my-gcp-project:claims_output_data_bucket.processed_claims',
                schema='ClaimID:STRING,VehicleID:STRING,ClaimDate:STRING,CasualCode:STRING,CustomerComplaint:STRING,TechnicianDiagnosis:STRING,RepairRecommendation:STRING,PartCosts:FLOAT64,LaborHours:FLOAT64,ClaimStatus:STRING',
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

if __name__ == '__main__':
    run()
