import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from pitch_dataset import PitchDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

ANNOTATIONS_FILE = "/home/amit/Documents/projects/pitch-detection/data/nsynth-train/examples.json"
AUDIO_DIR = "/home/amit/Documents/projects/pitch-detection/data/nsynth-train/audio"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE * 4

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_dataloader

def train_one_epoch(model, data_loader, loss_fn, optimizer, device):
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device).long()

        # calculate loss
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_one_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-----------------------")
    print("Training is done.")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # instantiating our dataset object and create data loader
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
    )

    pitch_dataset = PitchDataset(
        ANNOTATIONS_FILE, 
        AUDIO_DIR, 
        mel_spectrogram, 
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    train_dataloader = create_data_loader(pitch_dataset, BATCH_SIZE)

    # construct model and assign it to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # instantiate loss function + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(cnn, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), "cnnnet.pth")
    print("Model trained and stored at cnn.pth")
