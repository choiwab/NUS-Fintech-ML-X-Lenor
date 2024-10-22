# 🚀 Lenor AI Tutor - Finance Module Backend

Welcome to the **Finance Module Backend** of **Lenor AI Tutor**. Lenor is an AI-powered educational platform designed to deliver personalized tutoring experiences. This repository focuses on building the backend architecture to generate and evaluate financial learning conversations.

---

## 🌐 About Lenor AI Tutor

[Lenor AI Tutor](https://lenorai.com/) is an intelligent platform built to help students across various subjects, including finance. The AI tutor interacts with students through conversations that adapt to their learning styles, ensuring personalized support for each stage of their educational journey.

Check out Lenor on [LinkedIn](https://www.linkedin.com/company/lenor-eduai/posts/?feedView=all) to stay updated on their latest developments.

---

## 🔧 Backend Technology

In this project, the following tools and models to power our conversation generation and evaluation systems:

- **LLMs for conversation generation**:  
  We use **[Meta's LLaMA 3](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)** model to generate AI tutor-student dialogues, helping students in various learning stages.  

- **Conversation evaluation using Gemini API**:  
  The **[Google Gemini API](https://ai.google.dev/gemini-api?gad_source=1&gclid=Cj0KCQjwmt24BhDPARIsAJFYKk3D-svNN6QrnSpo_HxWDwg_yg6eOq624ALTX5x0uUyEHUAtJji46doaAvEnEALw_wcB&hl=ko)** is used for evaluating the quality and accuracy of AI-generated conversations. We assess the conversations based on predefined learning objectives and key stages.

  - **Conversation classificattion using Google Bart MNLI**:  
  The **[Facebook BART](https://huggingface.co/facebook/bart-large-mnli)** is used to classify the conversation data into 12 different learning stages and give scores based on zero-shot classification.

---

## 📚 Learning Stages in the Finance Module

The conversation generation process simulates interactions for the following **12 learning stages**:

1. Building Rapport with Students  
2. Assessing the Student's School Curriculum  
3. Evaluating the Student's Learning Style  
4. Quizzing Students for Current Knowledge  
5. Designing a Personalized Study Plan  
6. Beginning Lectures Following the Study Plan  
7. Explaining Concepts and Providing Examples  
8. Quizzing Students with Past Exam Questions  
9. Providing Images, Diagrams, and Online Resources  
10. Ensuring Students Can Ask Questions Anytime  
11. Updating Pedagogy According to Conversations  
12. Repeat for Next Topic According to Study Plan  

---

## 🧠 How It Works

### Conversation Generation

Using the **Meta-LLaMA 3** model, we generate dialogues between the AI tutor and the student based on their progress in one of the 12 learning stages. 

### Conversation Evaluation

Using the **Google Gemini API**, we evaluate these conversations to ensure they align with the learning objectives of each stage. The evaluation process checks for accuracy and relevance based on a set of predefined parameters for each stage.

### Classifying Conversations

We also implemented a zero-shot classification using Facebook's **BART-large MNLI** model to classify conversations into their respective learning stages. This helps us automatically categorize new conversation data.


## 🔍 Future Development

This project is still in progress, and we are actively working on:

- **Enhancing conversation quality with more LLM fine-tuning**
- **Adding more advanced evaluation metrics**
- **Expanding to other subjects beyond finance**
- **Improving the classification pipeline for real-time conversation analysis**

**Stay tuned for more updates! 🚧**


## 🌟 Acknowledgements
Special thanks to the NUS Fintech Society and the Lenor AI Tutor team for their support and collaboration on this project.
