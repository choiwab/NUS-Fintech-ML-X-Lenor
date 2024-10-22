import json
from transformers import pipeline

# Define the 12 learning stages
stages = [
    "Building Rapport with Students",
    "Assessing the Student's School Curriculum",
    "Evaluating the Student's Learning Style",
    "Quizzing Students for Current Knowledge",
    "Designing a Personalised Study Plan",
    "Beginning Lectures Following the Study Plan",
    "Explaining Concepts and Providing Examples",
    "Quizzing Students with Past Exam Questions",
    "Providing Images, Diagrams, and Online Resources",
    "Ensuring Students Can Ask Questions Anytime",
    "Updating Pedagogy According to Conversations",
    "Repeat for Next Topic According to Study Plan"
]

# Load the zero-shot classification model from transformers
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_conversation(conversation_text, stages):
    # Perform zero-shot classification
    result = classifier(conversation_text, stages)
    return result

# Load the JSON file with conversations
with open('conversations_output_new_short.json', 'r') as f:
    data = json.load(f)

# Store the results to be written to a new JSON file
output_results = []

# Iterate through each conversation in the dataset and classify it
for conversation_entry in data:
    conversation_text = " ".join(conversation_entry['output conversation'])  # Joining the conversation strings
    current_stage = conversation_entry['current learning stage']

    # Classify the conversation into one of the stages
    classification_result = classify_conversation(conversation_text, stages)

    # Create an output dictionary for this entry
    output_entry = {
        "actual conversation": conversation_text,
        "actual stage": current_stage,
        "predicted stage": classification_result['labels'][0],  # Top predicted stage
        "stages and scores": {label: score for label, score in zip(classification_result['labels'], classification_result['scores'])}
    }

    # Append the result to the list
    output_results.append(output_entry)

# Write the results to a new JSON file
output_file_path = 'classified_conversations_output.json'
with open(output_file_path, 'w') as output_file:
    json.dump(output_results, output_file, indent=4)

print(f"Results have been saved to {output_file_path}")
