import openai
import json
from tqdm import tqdm  # tqdm import
# conda activate llava

openai.api_key = YOUR_API_KEY
# Define paths for input and example files
example_file_path = "example_intentqa.txt"
input_file_path = "/mnt2/user25/Video_generation/LLoVi/data/intentqa/video/mdf_caption.json"
#input_file_path = "temp.json"
output_file_path = "intentqa.json"

# Load example text
with open(example_file_path, "r") as file:
    example_text = file.read()

# Load captions from input JSON
with open(input_file_path, "r") as file:
    captions_data = json.load(file)

# Function to generate summary for a given caption using GPT-3.5
def generate_summary(caption, example):
    prompt = f'''
You're a visual summary expert. You can accurately make a [SUMMARY] based on [CAPTION], where the [CAPTION] is textual descriptions of the video as seen from your first-person perspective.

[CAPTION]: Textual descriptions of first-person perspective videos, about natural human activities and behaviour. Each line represents a caption of video clip, each caption is separated by a semicolon, with a total of $duration lines describing 180 seconds of video. At the beginning of each caption, the #C indicates the image seen from your point of view, and the #O indicates the other people in the image you seen.
[SUMMARY]: Based on the CAPTIONS of these video clips, you need to summarise them into an overall description of the video, in chronological order.
I will give you an example as follow:
<Example>
{example}

Now, you should make a [SUMMARY] based on the [CAPTION] below. You SHOULD follow the format of example.
[CAPTION]
{caption}
[SUMMARY]
'''
    # Call the OpenAI API to generate the summary
    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",  # Use the appropriate model
        messages=[{"role": "system", "content": "You're a visual summary expert. You can accurately make a [SUMMARY] based on [CAPTION], where the [CAPTION] is textual descriptions of the video as seen from your first person perspective."},
                  {"role": "user", "content": prompt}],
        max_tokens=1500  # Adjust token limit based on your needs
    )
    # Extract the summary from the response
    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Generate summaries for all captions
summaries = {}
for video_id, captions in tqdm(captions_data.items(), desc="Generating Summaries"):
    combined_captions = "\n".join(captions)
    summary = generate_summary(combined_captions, example_text)
    summaries[video_id] = summary

# Save summaries to the output JSON file
with open(output_file_path, "w") as file:
    json.dump(summaries, file, indent=4)

print(f"Summaries saved to {output_file_path}")
