# AI-assited-claim-processing

## Claims Processing Pipeline with GCP

This repository contains code for historical data preparation, training the model using historical claim data, processing new claims using Google Cloud Functions and Apache Beam (Dataflow). The goal is to trigger a Dataflow job whenever a new claims file is uploaded to a specified Cloud Storage bucket. The Dataflow job performs data processing, feature extraction, and predictions using a pre-trained model.

## Setup and Deployment

1. **Data Preperation**
2. **Model Training**
3. **Deploy the Cloud Function**
4. **Run the Dataflow Pipeline**
5. **Upload Data to trigger the pipeline**
