import subprocess
import time

NUM_CLIENT = 2
NUM_ROUND = 5


def start_server():
    print("Starting Flower server...")
    # Replace 'flower_server.py' with the name of your server script
    subprocess.Popen(["python3", "server.py"])
    time.sleep(5)  # Give the server some time to start


def start_clients(num_clients):
    print(f"Starting {num_clients} Flower clients...")
    processes = []
    for i in range(num_clients):
        # Modify this command to use different client scripts or parameters if needed
        process = subprocess.Popen(["python3", "client.py", str(i)])
        processes.append(process)
        time.sleep(1)  # Optionally wait a bit between starting clients

    return processes


if __name__ == "__main__":
    start_server()
    client_processes = start_clients(NUM_CLIENT)  # Start clients (adjust the number as needed)

    # Optionally, wait for all clients to finish
    for process in client_processes:
        process.wait()
