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
            inputs = self.tokenizer(element[field], return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
            embeddings.append(embedding)

        # Concatenate structured data and embeddings
        combined_features = list(structured_data.values()) + embeddings.flatten().tolist()
        return [combined_features]


class PredictCostAndApproval(beam.DoFn):
    def __init__(self, project, model_name, version_name):
        super(PredictCostAndApproval, self).__init__()
        self.project = project
        self.model_name = model_name
        self.version_name = version_name
        self.prediction_client = discovery.build('ml', 'v1')

    def process(self, element):
        # Prepare the input for prediction
        input_data = {'instances': [element]}
        parent = f'projects/{self.project}/models/{self.model_name}/versions/{self.version_name}'

        # Call AI Platform Prediction
        request = self.prediction_client.projects().predict(name=parent, body=input_data)
        response = request.execute()

        # Extract predictions
        part_costs = response['predictions'][0]['part_costs']
        labor_hours = response['predictions'][0]['labor_hours']
        approval = response['predictions'][0]['approval']

        # Append predictions to the element
        element['PartCosts'] = part_costs
        element['LaborHours'] = labor_hours
        element['Approval'] = approval
        return [element]

def run(argv=None):
    pipeline_options = PipelineOptions(argv)
    with beam.Pipeline(options=pipeline_options) as p:
        (
            p
            | 'ReadClaimData' >> beam.io.ReadFromText('gs://your-bucket-name/claims-data.csv', skip_header_lines=1)
            | 'ParseCSV' >> beam.Map(lambda line: dict(zip(['ClaimID', 'VehicleID', 'ClaimDate', 'CasualCode', 'CustomerComplaint', 'TechnicianDiagnosis', 'RepairRecommendation'], line.split(','))))
            | 'ProcessClaims' >> beam.ParDo(ProcessClaims(bert_model_name='bert-base-uncased'))
            | 'PredictCostAndApproval' >> beam.ParDo(PredictCostAndApproval(project='your_project_id', model_name='your_model_name', version_name='your_version_name'))
            | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
                'your_project_id:your_dataset.your_table',
                schema='ClaimID:STRING,VehicleID:STRING,ClaimDate:STRING,CasualCode:STRING,CustomerComplaint:STRING,TechnicianDiagnosis:STRING,RepairRecommendation:STRING,PartCosts:FLOAT64,LaborHours:FLOAT64,Approval:STRING',
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

if __name__ == '__main__':
    run()
