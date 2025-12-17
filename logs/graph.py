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

file_name = "logs/isl_log.txt"
ref_file = "logs/ref_log.txt"

epochs, losses = read_file(file_name)
ref_epochs, ref_losses = read_file(ref_file)

plt.figure(figsize=(12, 7))
plt.plot(epochs, losses, 'b-', linewidth=2, label='Loss')
plt.plot(ref_epochs, ref_losses, 'r-', linewidth=2, label='Reference')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss over Epochs', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()

plt.figure(figsize=(12, 7))
losses = normalize_data(losses)
ref_losses = normalize_data(ref_losses)
plt.plot(epochs, losses, 'b-', linewidth=2, label='Loss')
plt.plot(ref_epochs, ref_losses, 'r-', linewidth=2, label='Reference')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss over Epochs (Normalized)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()