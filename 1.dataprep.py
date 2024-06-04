import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import torch

# Load structured data
data = pd.read_csv('historical_claims.csv')

import pandas as pd
from datetime import datetime

# Example DataFrames

vehicles_data = {
    'VehicleID': ['V1', 'V2', 'V3'],
    'ManufacturingDate': ['2018-01-01', '2019-06-15', '2020-11-10'],
    'VehicleType': ['Sedan', 'SUV', 'Truck']
}

repair_category_data = {
    'CasualCode': ['E01', 'M02', 'B03'],
    'RepairCategory': ['Engine', 'Mechanical', 'Battery']
}

# Create DataFrames
vehicles_df = pd.DataFrame(vehicles_data)
repair_category_df = pd.DataFrame(repair_category_data)

# Convert dates to datetime
data['ClaimDate'] = pd.to_datetime(data['ClaimDate'])
vehicles_df['ManufacturingDate'] = pd.to_datetime(vehicles_df['ManufacturingDate'])

# Merge data with vehicles_df to get VehicleType and ManufacturingDate
data = data.merge(vehicles_df, on='VehicleID', how='left')

# Calculate VehicleAge
data['VehicleAge'] = (data['ClaimDate'] - data['ManufacturingDate']).dt.days / 365

# Merge data with repair_category_df to get RepairCategory
data = data.merge(repair_category_df, on='CasualCode', how='left')

# Calculate time since last claim for the same vehicle
data = data.sort_values(by=['VehicleID', 'ClaimDate'])
data['TimeSinceLastClaim'] = data.groupby('VehicleID')['ClaimDate'].diff().dt.days

# Calculate ClaimFrequency for each vehicle up to the current claim date
data['ClaimFrequency'] = data.groupby('VehicleID').cumcount() + 1

# Example output
print(data)

structured_features = data[['VehicleAge', 'VehicleType', 'RepairCategory', 
                            'TimeSinceLastClaim', 'ClaimFrequency', 'CasualIssue']]

# Example text data
customer_complaints = data['CustomerComplaint']
technician_diagnosis = data['TechnicianDiagnosis']
repair_recommendations = data['RepairRecommendation']

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_embeddings(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy().flatten()
        embeddings.append(embedding)
    return torch.tensor(embeddings)

# Generate embeddings for text data
customer_complaint_embeddings = get_embeddings(customer_complaints)
technician_diagnosis_embeddings = get_embeddings(technician_diagnosis)
repair_recommendations_embeddings = get_embeddings(repair_recommendations)

# Sentiment Analysis using Google Cloud NLP
from google.cloud import language_v1

client = language_v1.LanguageServiceClient()

def get_sentiment_scores(texts):
    sentiment_scores = []
    for text in texts:
        document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)
        sentiment = client.analyze_sentiment(document=document).document_sentiment
        sentiment_scores.append(sentiment.score)
    return sentiment_scores

customer_complaint_sentiments = get_sentiment_scores(customer_complaints)

# Combine all features
features = torch.cat((structured_features, 
                      customer_complaint_embeddings, 
                      technician_diagnosis_embeddings, 
                      repair_recommendations_embeddings, 
                      torch.tensor(customer_complaint_sentiments).unsqueeze(1)), dim=1)

# Split features and targets for training
X = features.numpy()
y_costs = data['PartCosts'].values
y_hours = data['LaborHours'].values
y_labels = data['ClaimStatus'].values
