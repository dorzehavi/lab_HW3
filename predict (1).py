import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch.nn import functional as F
from dataset import HW3Dataset
import pandas as pd
from torch_geometric.data import DataLoader


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=8)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_features, hidden_dim),
                                  torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(torch.nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class EnsembleGNN(torch.nn.Module):
    def __init__(self, num_features, num_classes, models, hidden_dim):
        super(EnsembleGNN, self).__init__()
        self.models = nn.ModuleList(models)
        self.classifier1 = nn.Linear(len(models) * hidden_dim, hidden_dim)  # Classification layer 1
        self.classifier2 = nn.Linear(hidden_dim, hidden_dim)  # Classification layer 2
        self.classifier3 = nn.Linear(hidden_dim, num_classes)  # Classification layer 3

        self.batch_norm = nn.BatchNorm1d(hidden_dim * len(models))

        # Initialize linear layer parameters
        nn.init.xavier_uniform_(self.classifier1.weight)
        nn.init.zeros_(self.classifier1.bias)

    def forward(self, x, edge_index):

        out = torch.cat([model(x, edge_index) for model in self.models], dim=1)
        out = self.batch_norm(out)
        out = F.elu(out)
        out = self.classifier1(out)  # Pass through classification layer
        out = F.relu(out)
        out = F.dropout(out, p=0.4, training=self.training)  # Pass through classification layer
        out = self.classifier2(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.4, training=self.training)  # Pass through classification layer
        out = self.classifier3(out)
        return out


if __name__ == '__main__':
    # Load dataset
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset.data

    # Choose device
    device = torch.device('cpu')

    num_classes = 40
    # Add year data to features
    num_features = data.x.shape[1]
    data = data.to(device)
    hidden_dim = 256

    models = [
        GCN(num_features, num_classes, hidden_dim),
        GAT(num_features, num_classes, hidden_dim),
        GIN(num_features, num_classes, hidden_dim),
    ]

    model = EnsembleGNN(num_features, num_classes, models, hidden_dim)
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    model.to(device)
    model.eval()  # set model to evaluation mode

    # Create a DataLoader for making predictions in batches if your data is large
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predictions = []
    with torch.no_grad():  # no need to track gradients when making predictions
        for batch in loader:
            # Get the features and edge_index from the batch
            x, edge_index = batch.x.to(device), batch.edge_index.to(device)

            # Make predictions
            out = model(x, edge_index)

            # Get the predicted class for each sample in the batch
            pred = out.argmax(dim=1)  # no need to convert to numpy

            # Add the predictions to our list
            predictions.extend(pred.cpu().numpy())  # convert to numpy before extending the list

    # Save predictions to a CSV file
    df = pd.DataFrame({'idx': range(len(predictions)), 'prediction': predictions})
    df.to_csv('predictions.csv', index=False)