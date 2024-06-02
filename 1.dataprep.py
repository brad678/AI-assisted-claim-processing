import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel
import torch

# Load structured data
data = pd.read_csv('historical_claims.csv')

# Example structured features
structured_features = data[['PartCosts', 'LaborHours', 'VehicleAge']]

# Normalize structured features
scaler = StandardScaler()
normalized_structured_features = scaler.fit_transform(structured_features)

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
features = torch.cat((normalized_structured_features, 
                      customer_complaint_embeddings, 
                      technician_diagnosis_embeddings, 
                      repair_recommendations_embeddings, 
                      torch.tensor(customer_complaint_sentiments).unsqueeze(1)), dim=1)

# Split features and targets for training
X = features.numpy()
y_costs = data['PartCosts'].values
y_hours = data['LaborHours'].values
y_labels = data['ClaimStatus'].values
