import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ### Summary
# 1. **Input Channels (3)**: Correspond to the three text fields.
# 2. **Output Channels (16)**: Number of filters to extract diverse features.
# 3. **Convolutional Layer**: Extracts features, maintaining sequence length.
# 4. **Pooling Layer**: Reduces sequence length by half.
# 5. **Flattening**: Converts 3D tensor to 2D tensor for fully connected layers.
# 6. **Fully Connected Layers**: Process features and produce regression outputs.
# 7. **Fully Connected Layers**: Process features, along with regression outputs and produces classification output.

# Here’s the detailed transformation of the input data through the layers:

# 1. **Input Embeddings**: Shape `(batch_size, 3, 768)`
# 2. **After Convolution**: Shape `(batch_size, 16, 768)`
# 3. **After Pooling**: Shape `(batch_size, 16, 384)`
# 4. **After Flattening**: Shape `(batch_size, 16 * 384 = 6144)`
# 5. **After Fully Connected Layer**: Shape `(batch_size, 50)`
# 6. **Regression Outputs**: Two outputs of shape `(batch_size, 1)` for `PartCosts` and `LaborHours`.
# 7. **Classification Output**: Output of shape `(batch_size, 1)` for `ClaimStatus`.

#  includes a convolutional layer, a pooling layer, fully connected layers, and output layers for both regression and classification tasks.
class ClaimsModel(nn.Module):
    def __init__(self, num_structured_features):
        super(ClaimsModel, self).__init__()
        # Convolutional layer (input channels: 3, output channels: 16, kernel size: 3)
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # Max pooling layer (kernel size: 2, stride: 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # Fully connected layer for CNN features (input size: 16 * 384, output size: 50)
        self.fc1 = nn.Linear(16 * 384, 50)
        # Fully connected layer for combined features
        self.fc_combined = nn.Linear(50 + num_structured_features, 50)
        # Output layers for regression tasks
        self.fc2_part_costs = nn.Linear(50, 1)
        self.fc2_labor_hours = nn.Linear(50, 1)
        # Output layer for classification task
        self.fc_claim_status = nn.Linear(50 + 2, 2)  # Including the two predicted regression outputs

    # Processes embeddings through the CNN layers, combines them with structured data, and generates predictions for part costs, labor hours, and claim status
    def forward(self, x_embeddings, x_structured):
        # Apply convolutional layer followed by ReLU activation and pooling layer
        x_embeddings = self.pool(F.relu(self.conv1(x_embeddings)))
        # Flatten the output from the pooling layer
        x_embeddings = x_embeddings.view(-1, 16 * 384)
        # Apply fully connected layer to CNN features
        x_embeddings = F.relu(self.fc1(x_embeddings))
        # Concatenate CNN features with structured data
        x_combined = torch.cat((x_embeddings, x_structured), dim=1)
        # Apply fully connected layer to combined features
        x_combined = F.relu(self.fc_combined(x_combined))
        # Predict part costs and labor hours
        part_costs = self.fc2_part_costs(x_combined)
        labor_hours = self.fc2_labor_hours(x_combined)
        # Concatenate combined features with predicted regression outputs
        x_final = torch.cat((x_combined, part_costs, labor_hours), dim=1)
        # Apply classification layer
        claim_status_pred = self.fc_claim_status(x_final)
        return part_costs, labor_hours, claim_status_pred

# Example structured data
structured_data = torch.tensor([[0.5, 1.2, 3.4]], dtype=torch.float32)  # Example structured features

# Initialize the model
num_structured_features = structured_data.shape[1]
model = ClaimsModel(num_structured_features)

# Define loss functions and optimizer
criterion_regression = nn.MSELoss()
criterion_classification = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example training loop
num_epochs = 20
batch_size = 32

# Example dataset
embeddings = torch.rand((100, 3, 128))  # Example embeddings
structured_data = torch.rand((100, num_structured_features))
part_costs = torch.rand((100, 1))
labor_hours = torch.rand((100, 1))
claim_status = torch.randint(0, 2, (100,))

dataset = TensorDataset(embeddings, structured_data, part_costs, labor_hours, claim_status)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for batch in dataloader:
        x_embeddings, x_structured, y_part_costs, y_labor_hours, y_claim_status = batch
        
        optimizer.zero_grad()
        
        # Forward pass 
        part_costs_pred, labor_hours_pred, claim_status_pred = model(x_embeddings, x_structured)
        
        # Compute loss
        loss_regression = criterion_regression(part_costs_pred, y_part_costs) + criterion_regression(labor_hours_pred, y_labor_hours)
        loss_classification = criterion_classification(claim_status_pred, y_claim_status)
        loss = loss_regression + loss_classification
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'claims_model.pth')

# Apply the model to new data
part_costs_pred, labor_hours_pred, claim_status_pred = model(embeddings, structured_data)
print("Predicted Part Costs:", part_costs_pred)
print("Predicted Labor Hours:", labor_hours_pred)
print("Predicted Claim Status:", claim_status_pred)
