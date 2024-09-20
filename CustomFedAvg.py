import flwr as fl


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize cumulative variables for global averages
        self.global_loss = 0.0
        self.global_mse = 0.0
        self.global_r2 = 0.0
        self.global_accuracy = 0.0
        self.total_clients = 0
        self.rounds = 0

    def aggregate_evaluate(self, client_evaluations, _round, config):
        """
        Override the aggregate_evaluate method to aggregate metrics from all clients.
        This method collects evaluation results from clients for each round.
        """
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(client_evaluations, _round, config)
        return aggregated_loss, aggregated_metrics



