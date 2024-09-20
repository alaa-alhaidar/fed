import os
import csv
import flwr as fl
from CustomFedAvg import CustomFedAvg
from terminal import NUM_ROUND, NUM_CLIENT


class CustomFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_evaluation_metrics = {}
        self.csv_file_path = f"result/SERVER_{NUM_CLIENT}_{NUM_ROUND}_aggregated_loss.csv"  # Path to the CSV file
        self.round_counter = 1  # Round tracker

        # Create the CSV file and write the headers if it doesn't exist
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["round_num", "aggregated_loss","agg"])

    def aggregate_evaluate(self, fit_results, config, round_num):

        # Call the parent class method to handle default aggregation
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(fit_results, config, round_num)

        # Store the final evaluation metrics
        self.final_evaluation_metrics = aggregated_metrics

        # Print the aggregated loss and metrics
        print(f"Aggregated Loss: {aggregated_loss}")

        # Write the aggregated loss to the CSV file
        with open(self.csv_file_path, mode="a", newline="") as file:

            writer = csv.writer(file)
            writer.writerow([self.round_counter, aggregated_loss, aggregated_metrics])
            # Increment the round counter
            self.round_counter += 1
        # Return aggregated loss and metrics dictionary
        return aggregated_loss, aggregated_metrics

    def get_final_metrics(self):
        """
        Return the final evaluation metrics aggregated from clients.
        """
        return self.final_evaluation_metrics


import flwr as fl
from com.terminal import NUM_ROUND, NUM_CLIENT


def main():
    config = fl.server.ServerConfig(num_rounds=NUM_ROUND)

    # Define the custom strategy
    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )

    # Start the Flower server
    server = fl.server.start_server(
        server_address="localhost:8080",
        config=config,
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,  # Increase message size if needed
    )

if __name__ == "__main__":
    main()
