import json
import datetime
from huggingface_hub import InferenceClient
import os
import random
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()
access_token_1 = os.getenv("GEMINI_1")
genai.configure(api_key=access_token_1)

client = InferenceClient(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    token=os.getenv("HF_TOKEN"),
)

validator = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction='''
            You are a chat evaluator. You will be given a conversation history between a tutor and a student.
            You are given a list of questions (in a python dictionary format, with the list of questions as the values and the list of topics as the keys) to evaluate the conversation by, for each question, return 1 (if the answer is "yes") else, return 0
            return your answer in the following format:
            {
            <topic>:<response>,
            ...
            }
            ''',
            generation_config={"response_mime_type": "application/json"}
            )


# return a answer from 0 to 10, with 0 being the question is not answered by the conversation history at all and 10 being the question is fully answered by the conversation history.
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

sub_stages = {
    "Building Rapport with Students": {
        "Initiate Conversations": "Has the conversation initiated casual discussions to learn about the student's interests?",
        "Create Comfortable Environment": "Has the conversation created a comfortable and welcoming environment for the student?",
        "Ensure Approachability": "Is the conversation ensuring approachability and openness to questions at any time?",
        "Recognize Achievements": "Has the conversation recognized and celebrated the student's achievements, no matter how small?",
        "Use Names": "Does the conversation address the student by their name to build a personal connection?",
        "Active Listening": "Is the conversation actively listening to the student's feedback and experiences?",
        "Incorporate Humor": "Has the conversation incorporated humor to make the interaction enjoyable?"
    },
    "Assessing the Student's School Curriculum": {
        "Review Curriculum Guidelines": "Has the conversation reviewed the school's curriculum guidelines to identify key subjects?",
        "Identify Learning Gaps": "Has the conversation identified any gaps in the student's learning where they may need additional support?",
        "Align with Standards": "Does the conversation align the study plan with the student's learning standards and assessments?",
        "Teacher Input": "Has the conversation gathered input from the student's teachers regarding their performance?",
        "Discuss Future Goals": "Has the conversation discussed the student's future goals and how the curriculum supports them?",
        "Evaluate Subject Relevance": "Has the conversation evaluated the relevance of the current subjects to the student's interests?",
        "Adjust Curriculum": "Is the conversation using feedback from the student to adjust the curriculum as needed?",
        "Monitor Progress": "Is the conversation monitoring the student's progress in meeting curriculum expectations?"
    },
    "Evaluating the Student's Learning Style": {
        "Identify Learning Style": "Has the conversation identified the student's preferred learning style (visual, auditory, kinesthetic)?",
        "Observe Engagement": "Is the conversation observing which teaching methods engage the student most?",
        "Adapt Teaching Strategies": "Is the conversation adapting teaching strategies based on the student's learning style?",
        "Experiment with Techniques": "Has the conversation experimented with different techniques to see what works best for the student?",
        "Encourage Reflection": "Is the conversation encouraging the student to reflect on their learning preferences?",
        "Utilize Technology": "Is the conversation utilizing technology that caters to the student's learning style?",
        "Reassess Periodically": "Is the conversation reassessing the student's learning style periodically?",
        "Create Inclusive Strategies": "Has the conversation created inclusive strategies that address all learning styles?"
    },
    "Quizzing Students for Current Knowledge": {
        "Conduct Pre-Assessment": "Has the conversation conducted pre-assessment tests to gauge the student's knowledge?",
        "Engage in Discussions": "Has the conversation engaged in one-on-one discussions to encourage detailed responses?",
        "Identify Strengths and Weaknesses": "Has the conversation identified the student's strengths and weaknesses based on quizzes?",
        "Implement Quick Quizzes": "Is the conversation implementing quick quizzes after each topic to reinforce learning?",
        "Encourage Peer Quizzes": "Has the conversation encouraged the student to quiz their peers for collaborative learning?",
        "Provide Immediate Feedback": "Is the conversation providing immediate feedback on quiz performance for better understanding?",
        "Adjust Study Focus": "Has the conversation adjusted the study focus based on quiz results?"
    },
    "Designing a Personalised Study Plan": {
        "Set Clear Objectives": "Has the conversation set clear objectives for the student's study plan?",
        "Create Timetable": "Has the conversation created a timetable that allocates time for subjects based on the student's needs?",
        "Incorporate Varied Resources": "Is the conversation incorporating varied resources such as textbooks, online courses, and multimedia?",
        "Ensure Manageable Workload": "Is the conversation ensuring the workload is balanced and manageable for the student?",
        "Include Breaks": "Has the conversation included regular breaks to maintain focus and prevent burnout?",
        "Adjust as Needed": "Is the conversation adjusting the study plan based on the student's feedback and progress?",
        "Engage Student in Planning": "Has the conversation engaged the student in creating their own study plan?",
        "Monitor and Review": "Is the conversation monitoring and reviewing the student's progress regularly?"
    },
    "Beginning Lectures Following the Study Plan": {
        "Follow Timetable": "Is the conversation following the established timetable to cover all topics?",
        "Use Engaging Materials": "Is the conversation using engaging materials like videos and interactive elements?",
        "Encourage Participation": "Is the conversation encouraging the student to participate and ask questions during lectures?",
        "Summarize Key Points": "Is the conversation regularly summarizing key points to reinforce learning?",
        "Connect to Real World": "Is the conversation connecting lessons to real-world examples for better understanding?",
        "Provide Clear Instructions": "Is the conversation providing clear instructions for what is expected during each session?",
        "Adjust Pacing": "Is the conversation adjusting the pace of the lesson based on the student's comprehension?",
        "Use Quick Assessments": "Is the conversation using quick assessments during lectures to gauge understanding?"
    },
    "Explaining Concepts and Providing Examples": {
        "Break Down Complex Ideas": "Is the conversation breaking down complex ideas into manageable pieces?",
        "Use Analogies": "Is the conversation using analogies to relate new information to familiar ideas?",
        "Use Visual Aids": "Is the conversation incorporating visual aids like diagrams and charts to explain concepts?",
        "Provide Context": "Is the conversation providing context for why a concept is important and how it applies to real life?",
        "Encourage Questions": "Is the conversation encouraging the student to ask questions for clarification?",
        "Provide Multiple Examples": "Is the conversation offering multiple examples to cater to different learning styles?",
        "Connect to Previous Learning": "Is the conversation connecting new material to what the student has already learned?",
        "Check Understanding": "Is the conversation checking for understanding through follow-up questions or quizzes?"
    },
    "Quizzing Students with Past Exam Questions": {
        "Use Past Exam Questions": "Is the conversation using authentic past exam questions to familiarize the student with the format?",
        "Simulate Exam Conditions": "Is the conversation simulating exam conditions by conducting timed quizzes?",
        "Review Incorrect Answers": "Is the conversation reviewing incorrect answers with the student to improve understanding?",
        "Discuss Exam Strategies": "Is the conversation discussing exam strategies to help the student handle exam questions effectively?",
        "Track Progress": "Is the conversation tracking the student's progress over time on past exam questions?",
        "Provide Feedback": "Is the conversation providing constructive feedback on the student's quiz performance?",
        "Adapt Study Sessions": "Is the conversation adapting the study sessions based on the student's quiz outcomes?"
    },
    "Providing Images, Diagrams, and Online Resources": {
        "Use Visual Materials": "Is the conversation incorporating visual learning materials like infographics and images?",
        "Share Online Resources": "Is the conversation sharing online resources like reputable websites and online courses for further study?",
        "Recommend Educational Videos": "Is the conversation recommending educational videos that explain complex topics?",
        "Create Resource Bank": "Has the conversation created a resource bank for the student to access at any time?",
        "Teach Digital Literacy": "Is the conversation teaching the student how to search for and evaluate online information?",
        "Recommend Interactive Tools": "Is the conversation recommending apps and tools that promote interactive learning?",
        "Use Diagrams in Lectures": "Is the conversation including diagrams in lectures to reinforce concepts?",
        "Offer Supplementary Materials": "Is the conversation offering supplementary materials for students who want to delve deeper?"
    },
    "Ensuring Students Can Ask Questions Anytime": {
        "Establish Q&A": "Has the conversation established a Q&A period at the end of each session?",
        "Check for Understanding": "Is the conversation regularly checking for understanding by asking if the student has questions?",
        "Create Safe Space": "Has the conversation created a safe space where all questions are valid and important?",
        "Follow Up on Questions": "Is the conversation following up on any unanswered questions in subsequent sessions?"
    },
    "Updating Pedagogy According to Conversations": {
        "Hold Feedback Sessions": "Is the conversation holding regular feedback sessions to discuss the student's learning experiences?",
        "Adapt Teaching Methods": "Is the conversation adapting teaching methods based on student feedback?",
        "Incorporate Suggestions": "Is the conversation incorporating student suggestions to enhance the learning environment?",
        "Stay Informed": "Is the conversation staying informed on best practices and applying effective teaching methodologies?",
        "Monitor Engagement": "Is the conversation monitoring the student's engagement levels to adjust teaching practices?",
        "Reflect on Teaching": "Is the conversation regularly reflecting on the effectiveness of teaching strategies?",
        "Encourage Growth Mindset": "Is the conversation encouraging a growth mindset by being open to learning from feedback?",
        "Continuous Improvement": "Is the conversation continuously improving teaching approaches based on ongoing adjustments?"
    },
    "Repeat for Next Topic According to Study Plan": {
        "Repeat for Next Topic": "Has the conversation completed all of the following: Building Rapport with Students, Assessing the Student's School Curriculum, Evaluating the Student's Learning Style, Quizzing Students for Current Knowledge, Designing a Personalised Study Plan, Beginning Lectures Following the Study Plan, Explaining Concepts and Providing Examples, Quizzing Students with Past Exam Questions, Providing Images, Diagrams, and Online Resources, Ensuring Students Can Ask Questions Anytime, Updating Pedagogy According to Conversations?"
    }
}

sub_stages_combined = {}
for i in sub_stages.values():
    sub_stages_combined.update(i)

def generate_conversations(num_conversations, file_name):
    data = []
    error_count = 0

    for i in range(num_conversations):
        try:
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            # no_of_stages = [0] + random.sample(range(1, 12), random.randint(0,11))
            # curr_stages = [stages[i] for i in no_of_stages]
            # last_stage = curr_stages[-1]
            curr_stage = stages[random.randint(0,11)]

            # conversation_output = []

            # for stage in curr_stages:
            prompt = f"""We are now at the stage: "{curr_stage}".
            Please simulate the rest of the conversation between the Finance AI tutor and a student in this stage.
            Write the conversation in a human and informal way. Label each line as either the "AI tutor" or the "Student".
            Do not explicitly mention the name of the stage in the conversation and the make the conversation 3-4 lines.
            Do not include any additional text other than the labelling and the conversation itself
            """

            conversation = ""

            for message in client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=1.2,
                top_p=0.3,
                stream=True
            ):
                conversation += message.choices[0].delta.content

            conversation = conversation.split("\n\n")

            # conversation_output += conversation


            response = validator.generate_content(f'''
                Conversation History: {conversation}

                Questions to ask: {sub_stages_combined}
            ''')

            scores = json.loads(response.text)
            combined_scores = dict.fromkeys(stages, None)

            for stage in sub_stages:
                keys = list(sub_stages[stage].keys())
                combined_scores[stage] = sum([int(scores[i]) for i in keys]) / len(keys)

            data.append({
                "LLM model": "Meta-Llama-3-8B-Instruct",
                "Date": date_str,
                "Time": time_str,
                # "all stages": curr_stages,
                "current learning stage": curr_stage,
                "output conversation": conversation,
                **scores,
                "score by stage": combined_scores
            })
        except Exception as e:
            print("Oh no! Error occured: ", e)
            error_count += 1
        else:
            print("Yay! One new datapoint generated.")

    with open(file_name, mode='r') as file:
        file_data = json.load(file)
        data += file_data

    with open(file_name, mode='w') as file:
        json.dump(data, file, indent=4)

    return num_conversations - error_count

count = generate_conversations(300, 'conversations_output_new_new.json')

print(f"{count} conversations stored in 'conversations_output_new_new.json'")
