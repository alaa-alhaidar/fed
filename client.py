import random
import csv
import os
import torch
import torch.nn as nn
import flwr as fl
from sklearn.metrics import mean_squared_error, r2_score
import time
import numpy as np
from model import Net
from linearModel import LinearModel
from terminal import NUM_ROUND, NUM_CLIENT
client_counter =  random.randint(0, 1000)
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def generate_client_id():
    global client_counter
    client_counter += 1
    return client_counter - 1  # Return the current value before incrementing


def get_data_loaders(client_id, num_clients, batch_size, min_samples=500, max_samples=8000):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Shuffle dataset indices for train set
    train_indices = np.arange(len(train_dataset))
    np.random.shuffle(train_indices)

    # Shuffle dataset indices for test set
    test_indices = np.arange(len(test_dataset))
    np.random.shuffle(test_indices)

    # Determine sample size for this client (train set)
    total_train_samples = len(train_dataset)
    min_samples_for_client = max(min_samples, total_train_samples // (num_clients * 2))
    max_samples_for_client = min(max_samples, total_train_samples // (num_clients // 2))
    num_client_train_samples = np.random.randint(min_samples_for_client, max_samples_for_client + 1)

    # Calculate start and end indices for train set
    start_index_train = np.random.randint(0, max(1, total_train_samples - num_client_train_samples))
    end_index_train = start_index_train + num_client_train_samples

    client_train_indices = train_indices[start_index_train:end_index_train]

    # Create train subset and loader
    train_subset = Subset(train_dataset, client_train_indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Determine sample size for this client (test set)
    total_test_samples = len(test_dataset)
    min_test_samples_for_client = max(min_samples // 2, total_test_samples // (num_clients * 2))  # Smaller test set
    max_test_samples_for_client = min(max_samples // 2, total_test_samples // (num_clients // 2))

    num_client_test_samples = np.random.randint(min_test_samples_for_client, max_test_samples_for_client + 1)

    # Calculate start and end indices for test set
    start_index_test = np.random.randint(0, max(1, total_test_samples - num_client_test_samples))
    end_index_test = start_index_test + num_client_test_samples

    client_test_indices = test_indices[start_index_test:end_index_test]

    # Create test subset and loader
    test_subset = Subset(test_dataset, client_test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    # Debugging: Print information about the datasets
    print(f"Client ID: {client_id}")
    print(f"Training indices example of 10: {client_train_indices[:10]}")  # Print the first 10 indices for debugging
    print(f"Number of training samples: {len(train_subset)}")
    print(f"Test indices example of 10: {client_test_indices[:10]}")  # Print the first 10 indices for debugging
    print(f"Number of test samples: {len(test_subset)}")
    print(f"Start index (train): {start_index_train}, End index (train): {end_index_train}")
    print(f"Start index (test): {start_index_test}, End index (test): {end_index_test}")

    return train_loader, test_loader



# Define the Flower client class
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        # Initialize cumulative variables for averaging
        self.cumulative_loss = 0.0
        self.cumulative_accuracy = 0.0
        self.cumulative_mse = 0.0
        self.cumulative_r2 = 0.0
        self.eval_count = 0  # Track the number of evaluation rounds
        self.is_final_round = True
        self.NUM_ROUND = NUM_ROUND
        self.client_id=client_id



    def get_parameters(self, config=None):
        return [val.cpu().detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, model_param in zip(parameters, self.model.parameters()):
            model_param.data = torch.tensor(param, dtype=model_param.dtype).to(self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        start_time = time.time()  # Record start time
        self.train()
        end_time = time.time()  # Record end time
        training_time = end_time - start_time  # Calculate training time
        #print(f"Training time: {training_time:.2f} seconds")  # Print training time
        return self.get_parameters(), len(self.train_loader.dataset), {}




    def evaluate(self, parameters, config):
        # Set parameters for this round
        self.set_parameters(parameters)

        # Perform evaluation and get metrics
        loss, accuracy, mse, r2 = self.test()

        # Update cumulative metrics
        self.cumulative_loss += loss
        self.cumulative_accuracy += accuracy
        self.cumulative_mse += mse
        self.cumulative_r2 += r2
        self.eval_count += 1

        # If it's the final round, compute the final averages and save to CSV
        if self.NUM_ROUND == NUM_ROUND:
            final_avg_loss = self.cumulative_loss / self.eval_count
            final_avg_accuracy = self.cumulative_accuracy / self.eval_count
            final_avg_mse = self.cumulative_mse / self.eval_count
            final_avg_r2 = self.cumulative_r2 / self.eval_count

            # Print final averaged metrics for the entire training
            print(f"Client {self.client_id} Final Averaged Client Metrics (Round {self.eval_count}): "
                  f"Loss = {final_avg_loss}, Accuracy = {final_avg_accuracy}, MSE = {final_avg_mse}, R2 = {final_avg_r2}")

            # Create a dictionary for the final averaged metrics
            final_metrics = {"mse": final_avg_mse, "r2": final_avg_r2}

            # Save final metrics to CSV
            self.save_metrics_to_csv(final_avg_loss, final_avg_accuracy, final_avg_mse, final_avg_r2)

            # Return the final averaged loss, accuracy, and additional metrics
            return final_avg_loss, 1, final_metrics

        # Return metrics for intermediate rounds
        return loss, 1, {"mse": mse, "r2": r2}

    def save_metrics_to_csv(self, loss, accuracy, mse, r2):
        """Save the final averaged metrics to a shared CSV file."""
        csv_file = f"result/{NUM_CLIENT}_{NUM_ROUND}_final_metrics.csv"  # Shared CSV file

        # Check if the file exists; if not, create it and add the header
        file_exists = os.path.isfile(csv_file)

        # Open the CSV file in appended mode
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)

            # If the file does not exist, write the header first
            if not file_exists:
                writer.writerow(['Client ID', 'Loss', 'Accuracy', 'MSE', 'R2'])

            # Write the final averaged metrics for this client
            writer.writerow([self.client_id, loss, accuracy, mse, r2])

        print(f"Metrics for Client {self.client_id} saved to {csv_file}")

    def set_final_round(self, is_final_round):
        """Set a flag indicating that this is the final evaluation round."""
        self.is_final_round = is_final_round

    def train(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Collect true and predicted labels for MSE and R² calculation
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = correct / total
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return loss / total, accuracy, mse, r2


# Set up the device, model, and data loaders
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_clients = 10  # Example number of clients

# Automatically assign a client ID
client_id = generate_client_id()

# choose the model for training
model = Net().to(DEVICE)  # Net(), LinearModel()
train_loader, test_loader = get_data_loaders(client_id, num_clients, batch_size=32)

# Start the Flower client using the new API
client = FlowerClient(model, train_loader, test_loader, DEVICE)

# Updated code: use start_client() instead of start_numpy_client()
fl.client.start_client(server_address="localhost:8080", client=client.to_client())
