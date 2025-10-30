# %%
import os
import argparse
from client import DIVASClient
import flwr as fl
import torch

import logging
import warnings
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("flwr").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

# %%
parser = argparse.ArgumentParser(description="Federated Client")
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs for training')
parser.add_argument('--client_id', type=str, default='client_1',
                    help='Number of the client (default: client_1)')
parser.add_argument('--data_path', type=str, default='./run_txt',
                    help='Path to the dataset directory')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Device to run the computations on (default: cuda:0)')
parser.add_argument('--server_address', type=str, default='localhost:39705', 
                    help='Server address for Flower')


# args = parser.parse_args(args=[])
args = parser.parse_args()

torch.backends.cudnn.benchmark = True  
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
torch.cuda.empty_cache()
# %%
if __name__ == "__main__":

    # fl.client.start_client(
    fl.client.start_numpy_client(
        server_address="localhost:39705",
        client=DIVASClient(args.client_id, args.data_path, args)
    )
