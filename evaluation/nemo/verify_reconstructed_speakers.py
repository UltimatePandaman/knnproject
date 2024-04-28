import os
print("Loading NeMo library, this may take a while...")
import nemo.collections.asr as nemo_asr
import torch

# Load the speaker verification model
print("Loading speaker verification model, this may take a while...")
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")
print("Speaker verification model loaded.")

def verify_speakers(reference_path, reconstructed_path):
    if reference_path == reconstructed_path:
        return 1.0
    # Calculate similarity score
    #SRC: https://github.com/NVIDIA/NeMo/blob/cdf3266ef98d8e91ea0b004feca089c6696d9ea7/nemo/collections/asr/models/label_models.py#L562
    #Get embeddings
    embs1 = speaker_model.get_embedding(reference_path).squeeze()
    embs2 = speaker_model.get_embedding(reconstructed_path).squeeze()
    #Length Normalize
    X = embs1 / torch.linalg.norm(embs1)
    Y = embs2 / torch.linalg.norm(embs2)
    # Score
    similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
    similarity_score = (similarity_score + 1) / 2

    return similarity_score

## Load Data
print(f"Start testing")

speakers = ["p225", "p226", "p227", "p228", "p232", "p243"]
target_file = "verification_results.csv"

if os.path.exists(target_file):
    os.remove(target_file)

with open(target_file, "a") as f:
    f.write(f"ref_file,gen_file,score\n")

for speaker in speakers:
    reference_directory = os.path.join("..", "data", "data", speaker + "-test")
    reference_wav = os.path.join(reference_directory, os.listdir(reference_directory)[0])
    generated_directory = os.path.join("..", "data", "converted_sound", speaker)

    for file in os.listdir(generated_directory):
        gen_path = os.path.join(generated_directory, file)
        print(f"Running {speaker}/{file} test")
        
        # Iterate over all combinations of reference and reconstructed files
        print("Verifying speakers...")

        # Verify the speakers
        score = verify_speakers(reference_wav, gen_path)
        # Save the result to a log file
        with open(target_file, "a") as f:
            f.write(f"{reference_wav},{gen_path},{score}\n")