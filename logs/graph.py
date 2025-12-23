import matplotlib.pyplot as plt
import numpy as np

def read_file(filename):
    epochs = []
    losses = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    epochs.append(float(parts[0]))
                    losses.append(float(parts[1]))
    return np.array(epochs), np.array(losses)

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

file_name = "log.txt"

epochs, losses = read_file(file_name)

plt.figure(figsize=(12, 7))
plt.plot(epochs, losses, 'b-', linewidth=2, label='Loss')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss over Epochs', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("graph.png")