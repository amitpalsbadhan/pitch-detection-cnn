import torch
import torchaudio
from cnn import CNNNetwork
from pitch_dataset import PitchDataset
from train import SAMPLE_RATE, NUM_SAMPLES

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']

def pitch_class(midi_num: int) -> int:
    """Return 0…11 for C…B."""
    return midi_num % 12

def pitch_name(midi_num: int) -> str:
    """Return the note name (ignoring octave)."""
    return NOTE_NAMES[pitch_class(midi_num)]

ANNOTATIONS_FILE = "/home/amit/Documents/projects/pitch-detection/data/nsynth-test/examples.json"
AUDIO_DIR = "/home/amit/Documents/projects/pitch-detection/data/nsynth-test/audio"

class_mapping = [str(midi) for midi in range(21, 21 + 200)]

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected

if __name__ == "__main__":
    # load back the model
    cnn = CNNNetwork()
    state_dict = torch.load("cnnnet.pth")
    cnn.load_state_dict(state_dict)

    # load pitch validation dataset
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
        "cpu",
    )

    exact_matches = 0
    class_matches = 0
    total = 4000

    for i in range(total):
        inp, target = pitch_dataset[i][0], pitch_dataset[i][1]
        inp = inp.unsqueeze(0)

        predicted_str, expected_str = predict(cnn, inp, target, class_mapping)
        pred_midi = int(predicted_str)
        exp_midi  = int(expected_str)

        # Exact match?
        if pred_midi == exp_midi:
            exact_matches += 1

        # Same note class?
        if pitch_class(pred_midi) == pitch_class(exp_midi):
            class_matches += 1
        else:
            print("DIFFERENT CLASS",
                  f"pred={pitch_name(pred_midi)}({pred_midi}),",
                  f"exp={pitch_name(exp_midi)}({exp_midi})")

    print(f"Exact accuracy:       {exact_matches/total:.2%}")
    print(f"Pitch‑class accuracy: {class_matches/total:.2%}")