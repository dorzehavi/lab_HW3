import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch.nn import functional as F
from dataset import HW3Dataset
import os


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


# Define train function
def train(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)  # Add edge_index here
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


# Define test function
def test(model, data, mask):
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).max(dim=1)[1]  # Add edge_index here

    correct = preds[mask] == data.y[mask]

    acc = int(correct.sum()) / len(correct)
    return acc


if __name__ == '__main__':

    # Load dataset
    dataset = HW3Dataset(root='data/hw3/')
    data = dataset.data
    data.y = data.y.squeeze(1)  # squeeze out the extra dimension

    # Choose device
    device = torch.device('cpu')

    num_classes = data.y.max().item() + 1
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
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # Set path for saving models
    save_path = './saved_models/'
    os.makedirs(save_path, exist_ok=True)

    # Initialize best validation accuracy
    best_val_acc = 0.0

    # Train ensemble model
    print('Starting training for GAT model')
    for epoch in range(1000):
        loss = train(model, data, optimizer, data.train_mask)
        train_acc = test(model, data, data.train_mask)
        val_acc = test(model, data, data.val_mask)
        print(f'Epoch: {epoch + 1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        # If the validation accuracy of the current model is better than the previous best, save the model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pt'))
            print(f'Saving model at epoch {epoch+1} with validation accuracy {val_acc:.4f}')