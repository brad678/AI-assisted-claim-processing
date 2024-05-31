import torch.nn as nn
import torch.nn.functional as F

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
        self.fc_classification = nn.Linear(50 + 2, 2)  # Including the two predicted regression outputs
    
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
        classification_output = self.fc_classification(x_final)
        return part_costs, labor_hours, classification_output

# Example structured data
structured_data = torch.tensor([[0.5, 1.2, 3.4]], dtype=torch.float32)  # Example structured features

# Initialize the model
num_structured_features = structured_data.shape[1]
model = ClaimsModel(num_structured_features)

# Apply the model
part_costs_pred, labor_hours_pred, classification_output = model(embeddings, structured_data)
print("Predicted Part Costs:", part_costs_pred.item())
print("Predicted Labor Hours:", labor_hours_pred.item())
print("Classification Output:", classification_output)
