print(f"Importing libs")
import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline # type: ignore

def convert_numbers_to_words(text):
    # Dictionary mapping digits to their word equivalents
    numbers_to_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
    }
    
    # Replace numbers with their word equivalents
    words = []
    for word in text.split():
        if word.isdigit():
            word = numbers_to_words.get(word, word)
        words.append(word)
    
    # Assemble the converted text
    converted_text = ' '.join(words)
    return converted_text

def calculate_similarity(content1, content2):
    # Convert numbers to words
    content1 = convert_numbers_to_words(content1)
    content2 = convert_numbers_to_words(content2)
    
    # Tokenize text into words
    words1 = content1.split()
    words2 = content2.split()
    
    # Calculate the number of common words
    num_common_words = len(set(words1) & set(words2))
    
    # Calculate the total number of words as the maximum of words in both texts
    total_words = max(len(words1), len(words2))
    
    # Calculate the percentage similarity
    similarity_percentage = (num_common_words / total_words) * 100
    return similarity_percentage

## SRC: https://huggingface.co/openai/whisper-large-v3
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading whisper model")
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

## Load Data
print(f"Start testing")

speakers = ["p233", "p254", "p256"]
target_file = "transcribe_results.csv"

if os.path.exists(target_file):
    os.remove(target_file)

with open(target_file, "a") as f:
    f.write(f"ref_file,ref_content,gen_file,gen_content,score\n")

for speaker in speakers:
    reference_directory = os.path.join("..", "data", "data", speaker + "-val")
    generated_directory = os.path.join("..", "data", "converted_sound", speaker)

    for file in os.listdir(reference_directory):
        print(f"Running test on {speaker}/{file}")

        ref_path = os.path.join(reference_directory, file)
        gen_path = os.path.join(generated_directory, file)
        if not os.path.isfile(ref_path) or not os.path.isfile(gen_path):
            print(f"Missing WAV files for {speaker}/{file}. Skipping...")
            continue

        # Transcription for source.wav
        result_ref = pipe(ref_path, generate_kwargs={"language": "english"})
        ref_content = result_ref['text'].replace(",", "").replace(".", "").lower()
        print(f"Transcription for {speaker}/{file} ref was completed.")

        # Transcription for new.wav
        result_gen = pipe(gen_path, generate_kwargs={"language": "english"})
        gen_content = result_gen['text'].replace(",", "").replace(".", "").lower()
        print(f"Transcription for {speaker}/{file} gen was completed.")

        # Calculate similarity
        similarity = calculate_similarity(ref_content, gen_content)

        with open(target_file, "a") as f:
            f.write(f"{ref_path},{ref_content},{gen_path},{gen_content},{round(similarity, 5)}\n")
        
        if similarity == 100:
            print(f"Transcription for {speaker}/{file} matches the target file.")
        else:
            print(f"Transcription for {speaker}/{file} does not match the target file, achieved similarity of {round(similarity, 2)}%.")
