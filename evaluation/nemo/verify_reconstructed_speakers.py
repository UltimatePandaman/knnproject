import os
print("Loading NeMo library, this may take a while...")
import nemo.collections.asr as nemo_asr
import torch

# Load the speaker verification model
print("Loading speaker verification model, this may take a while...")
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
print("Speaker verification model loaded.")

def get_speaker_data(path):
    embs = speaker_model.get_embedding(path).squeeze()
    vector = embs / torch.linalg.norm(embs)
    return vector

def verify_speakers(reference_data, gen_path):
    # Calculate similarity score
    #SRC: https://github.com/NVIDIA/NeMo/blob/cdf3266ef98d8e91ea0b004feca089c6696d9ea7/nemo/collections/asr/models/label_models.py#L562

    #Get data
    X = reference_data
    Y = get_speaker_data(gen_path)

    # Score
    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2

    return similarity_score

## Load Data
print(f"Start testing")

speakers = ["p225", "p226"]
target_file = "verification_results.csv"

if os.path.exists(target_file):
    os.remove(target_file)

with open(target_file, "a") as f:
    f.write(f"ref_file,gen_file,score\n")

for speaker in speakers:
    reference_directory = os.path.join("..", "project_data", "source", speaker)
    reference_wav = os.path.join(reference_directory, os.listdir(reference_directory)[0])
    generated_directory = os.path.join("..", "project_data", "converted_sound", speaker)

    reference_data = get_speaker_data(reference_wav)

    for file in os.listdir(generated_directory):
        gen_path = os.path.join(generated_directory, file)
        print(f"Running {speaker}/{file} test")
        
        # Iterate over all combinations of reference and reconstructed files
        print("Verifying speakers...")

        # Verify the speakers
        score = verify_speakers(reference_data, gen_path)

        # Save the result to a log file
        with open(target_file, "a") as f:
            f.write(f"{reference_wav},{gen_path},{score}\n")