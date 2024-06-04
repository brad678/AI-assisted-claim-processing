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

def merge_data(claim_data, vehicle_data, repair_data):
    """Merges data from three PCollections based on common keys."""
    # Convert dates to datetime
    claim_data['ClaimDate'] = pd.to_datetime(claim_data['ClaimDate'])
    vehicle_data['ManufacturingDate'] = pd.to_datetime(vehicle_data['ManufacturingDate'])
    
    # Merge data with vehicle_data to get VehicleType and ManufacturingDate
    claim_data = claim_data.merge(vehicle_data, on='VehicleID', how='left')
    
    # Calculate VehicleAge
    claim_data['VehicleAge'] = (claim_data['ClaimDate'] - claim_data['ManufacturingDate']).dt.days / 365
    
    # Merge data with repair_data to get RepairCategory
    claim_data = claim_data.merge(repair_data, on='CasualCode', how='left')
    
    # Calculate time since last claim for the same vehicle
    claim_data = claim_data.sort_values(by=['VehicleID', 'ClaimDate'])
    claim_data['TimeSinceLastClaim'] = claim_data.groupby('VehicleID')['ClaimDate'].diff().dt.days
    
    # Calculate ClaimFrequency for each vehicle up to the current claim date
    claim_data['ClaimFrequency'] = claim_data.groupby('VehicleID').cumcount() + 1

def run(argv=None):
    pipeline_options = PipelineOptions(argv)
    with beam.Pipeline(options=pipeline_options) as p:
        # Read claim data
        claim_data = (
            p
            | 'ReadClaimData' >> beam.io.ReadFromText('gs://claims_data_bucket/new_claim_data.csv', skip_header_lines=1)
            | 'ParseClaimCSV' >> beam.Map(lambda line: dict(zip(['ClaimID', 'VehicleID', 'ClaimDate', 'CasualCode', 'CasualIssue', 'CustomerComplaint', 'TechnicianDiagnosis', 'RepairRecommendation'], line.split(','))))
        )

        # Read vehicle data
        vehicle_data = (
            p
            | 'ReadVehicleData' >> beam.io.ReadFromText('gs://claims_data_bucket/vehicle_data.csv', skip_header_lines=1)
            | 'ParseVehicleCSV' >> beam.Map(lambda line: dict(zip(['VehicleID', 'ManufacturingDate', 'VehicleType'], line.split(','))))
        )

        # Read customer data
        repair_data = (
            p
            | 'ReadCustomerData' >> beam.io.ReadFromText('gs://claims_data_bucket/repair_data.csv', skip_header_lines=1)
            | 'ParseCustomerCSV' >> beam.Map(lambda line: dict(zip(['CasualCode', 'RepairCategory'], line.split(','))))
        )

        # Merge data from all three PCollections
        merged_data = (
            p
            | 'MergeData' >> beam.Create([claim_data, vehicle_data, repair_data])
            | 'ApplyMergeFunction' >> beam.FlatMap(merge_data)
        )

        # Process claims
        (
            merged_data
            | 'PredictCostAndStatus' >> beam.ParDo(PredictCostAndStatus(project='my-gcp-project', model_name='claims_prediction_model', version_name='v1'))
            | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
                'my-gcp-project:claims_output_data_bucket.processed_claims',
                schema='ClaimID:STRING,VehicleID:STRING,ClaimDate:STRING,CasualCode:STRING,CustomerComplaint:STRING,TechnicianDiagnosis:STRING,RepairRecommendation:STRING,PartCosts:FLOAT64,LaborHours:FLOAT64,ClaimStatus:STRING',
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
        )

if __name__ == '__main__':
    run()
