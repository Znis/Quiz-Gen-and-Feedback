import os
os.environ["WANDB_DISABLED"] = "true"
import streamlit as st
import fitz  # PyMuPDF
import json
import os
import nltk
import time
import requests
import ast
import random
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from unsloth import FastLanguageModel
# from transformers import TextStreamer
# import torch
# import pandas as pd
# from unsloth.chat_templates import get_chat_template

# nltk.download('punkt')
# nltk.download('punkt_tab')

nltk.download('punkt_tab')


API_URL = "http://localhost:11434/api/chat"
# API_URL = "http://ollama-container:11434/api/chat"

# API_URL = os.getenv("API_URL", "http://localhost:11434/api/chat")
HEADERS = {"Content-Type": "application/json"}



### without streaming api call; full response at once; can comment if not in use
def get_response_from_api(question):
    data = {
        "model": "phi3_mcq",
        "messages": [{"role": "user", "content": question}],
        "stream": False,
        "options": {
            "temperature": 0.4,
            "repeat_penalty": 1.0,
            "seed": 3407,
            "top_k": 20,
            "top_p": 1.0,
        },
    }
    start_time=time.time()
    response = requests.post(API_URL, json=data, headers=HEADERS)
    time_taken=time.time()-start_time
    st.write(time_taken)
    print( time_taken)

    if response.status_code == 200:
        #st.write(response.json().get("message", {}).get("content", "No content"))
        return ast.literal_eval(response.json().get("message", {}).get("content", "No content"))
    else:
        return f"Error: {response.status_code}"


users = {
    "teacher1": {
        "password": "password1",
        "name": "Teacher 1",
        "role": "teacher"
    },
    "teacher2": {
        "password": "password2",
        "name": "Teacher 2",
        "role": "teacher"
    },
    "student1": {
        "password": "password3",
        "name": "Student 1",
        "role": "student"
    },
    "student2": {
        "password": "password4",
        "name": "Student 2",
        "role": "student"
    },
    "t": {
        "password": "t",
        "name": "Teacher",
        "role": "teacher"
    },
    "s": {
        "password": "s",
        "name": "Student",
        "role": "student"
    }
}


def check_login(username, password):
    user = users.get(username)
    if user and user['password'] == password:
        return user['name'],user['role']
    return None

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text("text")
    return text

# @st.cache_resource
# def load_model():
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name="results\model_gguf",  # Update this path
#         max_seq_length=2048,
#         dtype=None,
#         load_in_4bit=True,
#     )
#     tokenizer = get_chat_template(
#         tokenizer,
#         chat_template="phi-3",
#         mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
#     )
#     FastLanguageModel.for_inference(model)
#     return model, tokenizer

# model, tokenizer = load_model()



# def generate_quiz_from_context(context, num_questions=10):
#     try:
#         prompt = f"""
#         Generate {num_questions} Quizzes which require logical reasoning in a list of JSON format from the given context.
#         context: {context}
#         """
#         messages = [{"from": "human", "value": prompt}]
#         inputs = tokenizer.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=True,
#             return_tensors="pt"
#         ).to("cuda")
        
#         outputs = model.generate(input_ids=inputs, max_new_tokens=1024, use_cache=True)
#         generated_text = tokenizer.batch_decode(outputs)[0]
        
#         # Parse the generated text to extract the JSON quiz data
#         # You might need to implement custom parsing logic here
#         quiz_data = json.loads(generated_text)
        
#         return quiz_data
#     except Exception as e:
#         st.error(f"Error generating quiz: {str(e)}")
#         return None

    

def generate_explanation(context ,quiz_data_with_selected_options):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system", 
                "content": (
                    "Please generate explanation why the selected option is wrong or right based on the question and right option for the given context.\nThe explanation should include complete explanation and avoid cutoff mid-sentence.\nPlease rewrite the following explanation to be clear, concise, and no more than 4-5 sentences.\nUser will provide context, question and right option and selected option in the format as \"#context: particular context, #Question: particular question, #Right option: right option, #Selected option: selected option\". \n"
                )
            },
            {"role": "user", "content": f"#Question: {quiz_data_with_selected_options['question']}, #Right option: {quiz_data_with_selected_options['correct_option']}, #Selected option: {quiz_data_with_selected_options['selected_option']}"},
        ],
        max_tokens=150,  # Lowering the token limit to control length
        temperature=0.5,  # Lowering temperature for less randomness and more focus
        top_p=0.9  # Optional: slightly lower top_p for more coherent completions
    )

    return completion.choices[0].message.content

def generate_general_feedback(questions,score,time_elapsed):
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system", 
                "content":f"""
                     A user has completed a quiz with the following statistics:
    
                    - Total Questions: {questions}
                    - Correct Answers: {score}
                    - Time Taken: {time_elapsed} seconds # round up time value
    
                    Based on the user's performance, provide constructive feedback on their quiz attempt. 
                    Consider the user's score and the time taken. Offer encouragement and suggest areas for improvement.
                    Please keep feedback short and sweet.
                    """
            },
            {"role": "user", "content": f"#Questions:{questions},#Score:{score},#time taken:{time_elapsed},"},
        ],
        max_tokens=150,  # Lowering the token limit to control length
        temperature=0.5,  # Lowering temperature for less randomness and more focus
        top_p=0.9  # Optional: slightly lower top_p for more coherent completions
    )

    return completion.choices[0].message.content


def preview_quiz():
    st.header("Quiz Preview")
    # st.write(st.session_state.quiz_json)
    if 'topic' not in st.session_state.quiz_json.keys():
        topic = st.text_input("Title")
        if st.button("confirm"):
            st.session_state.quiz_json['topic'] = topic
            st.rerun()
    else:
        st.subheader(f"Topic: {st.session_state.quiz_json['topic']}")
        if st.button("Edit Title"):
            st.session_state.edit_question_index = -1
            st.session_state.is_editing = True
            st.rerun()
        # Iterate through the generated quiz questions
    for i, question_data in enumerate(st.session_state.quiz_json['questions']):
        # st.write(question_data)
        st.write(f"**Question {i + 1}:** {question_data['question']}")
        
        distractors = question_data['distractors']
        correct_option = question_data['correct_option']
        # options.append(question_data['correct_option'])
        # random.shuffle(options)
        
        # Display the options and highlight the correct one
        st.write(f"- ✅ **{correct_option}**")
        for option in distractors:
            st.write(f"- {option}")
        
        if st.button(f"Edit Question {i+1}", key=f"edit_{i}"):
            st.session_state.edit_question_index = i
            st.session_state.is_editing = True
            st.rerun()  # Move to edit mode for this question

def edit_quiz_question():
    i = st.session_state.edit_question_index
    # st.write(st.session_state.quiz_json)

    if i == -1:
        st.subheader("Editing Title")
        title_text = st.text_input("Title", value=st.session_state.quiz_json['topic'])

        if st.button("Save Changes"):
            st.session_state.quiz_json['topic'] = title_text
            
            # Exit editing mode
            st.session_state.is_editing = False
            st.success(f"Title updated successfully!")
            st.rerun()
    else:
        question_data = st.session_state.quiz_json['questions'][i]
        
        st.subheader(f"Editing Question {i + 1}")
        
        # Editable question input
        question_text = st.text_input("Question", value=question_data['question'])
        
        # Editable options with correct option selection
        distractors = [st.text_input(f"Distractor {j+1}", value=option) for j, option in enumerate(question_data['distractors'][:3])]
        correct_option = st.text_input("Correct Option", value=question_data['correct_option'])
    
        if st.button("Save Changes"):
            # Update the quiz data with edited question and options
            st.session_state.quiz_json['questions'][i]['question'] = question_text
            st.session_state.quiz_json['questions'][i]['distractors'] = distractors
            st.session_state.quiz_json['questions'][i]['right_option'] = correct_option
            # st.session_state.quiz_json['topic'] = title_text
            
            # Exit editing mode
            st.session_state.is_editing = False
            st.success(f"Question {i+1} updated successfully!")
            st.rerun()  # Refresh the dashboard to show the updated quiz preview

    if st.button("Cancel"):
        # Exit editing mode without saving
        st.session_state.is_editing = False
        st.rerun()

def finalize_quiz():
    col1, _, col2 = st.columns(3)
    with col1:
        if st.button("Cancel"):
            st.session_state.quiz_generated = False
            st.session_state.is_editing = False
            st.rerun()
    with col2:
        if st.button("Finalize Quiz"):
            # st.write(st.session_state.quiz_json)

            # Logic to save the quiz (e.g., write to a database or file)
            with open(f"quiz_{st.session_state.username}_{int(time.time())}.json", "w") as f:
                    json.dump(st.session_state.quiz_json, f)
                        
                    st.success("Quiz has been saved!")
            st.session_state.quiz_generated = False
            st.rerun()


def teacher_dashboard():
    _, col2, _ = st.columns([1, 3, 1])
    col2.title("Teacher's Dashboard")
    
    if "quiz_generated" not in st.session_state:
        st.session_state.quiz_generated = False
    if "generate_quizzes" not in st.session_state:
        st.session_state.generate_quizzes = False
    
    
    if not st.session_state.quiz_generated:

        if not st.session_state.generate_quizzes:
            _, col2, _ = st.columns([1, 3, 1])
            with col2:
                if st.button("Generate New Quizzes", use_container_width=True):
                    st.session_state.generate_quizzes = True
                    st.rerun()
            st.header("My Quizzes")
            quizzes = [f for f in os.listdir() if f.startswith(f"quiz_{st.session_state.username}") and f.endswith(".json")] 
            if len(quizzes) > 0:
                all_student_record_names = [f for f in os.listdir() if f.startswith("student_") and f.endswith(".json")]
                all_student_record = []
                all_quiz_record = []
                for student_record in all_student_record_names:
                    with open(student_record, "r") as f:
                        student_data = json.load(f)
                        all_student_record.append(student_data)
                for quiz in quizzes:
                    with open(quiz, "r") as f:
                        quiz_data = json.load(f)
                        all_quiz_record.append(quiz_data)
                # st.write(quizzes)
                for i, quiz in enumerate(all_quiz_record):
                    with st.expander(f"{i+1}. **{quiz['topic']}**"):
                        # st.write(quiz['questions'])
                        st.subheader(f"{i+1}. {quiz['topic']}")
                        st.write(f"**Questions: {len(quiz['questions'])}**")
                        stat_for_student = {}
                        for j, student_record in enumerate(all_student_record):
                            if quizzes[i] in student_record.keys():
                                if 'attempts' not in stat_for_student.keys():
                                # st.write(all_student_record_names[j])
                                # st.write(all_student_record_names[j].split('_')[1].split('.')[0])
                                    stat_for_student['student_name'] = [users[all_student_record_names[j].split('_')[1].split('.')[0]]['name']]
                                    stat_for_student['attempts'] = [len(student_record[quizzes[i]])]
                                    stat_for_student['best_score'] = [max([attempts['score'] for attempts in student_record[quizzes[i]]])]
                                else:
                                    stat_for_student['student_name'].append(users[all_student_record_names[j].split('_')[1].split('.')[0]]['name'])
                                    stat_for_student['attempts'].append(len(student_record[quizzes[i]]))
                                    stat_for_student['best_score'].append(max([attempts['score'] for attempts in student_record[quizzes[i]]]))
                        if len(stat_for_student) > 0:
                            _, col2, _ = st.columns([1,3,1])
                            with col2:
                                st.dataframe(stat_for_student)
                        else:
                            st.write("No student has attempted this quiz.")
                        # st.write(quiz['questions'][0]['question'])
                        
            else:
                st.write("No quiz found.")        
        else:
            _, col2, _ = st.columns([1, 3, 1])
            with col2:
                if st.button("View Generated Quizzes", use_container_width=True):
                    st.session_state.generate_quizzes = False
                    st.rerun()
            # Teachers generate quizzes here
            st.session_state.context = st.text_area("Enter context for quiz generation:", height=300)
            uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")
            number = st.number_input("Number of questions:", min_value=1, max_value=10, value=5)

            if st.button('Generate'):
                if not st.session_state.context and uploaded_file is None:
                    st.warning("Please enter some context or upload a .pdf file.")
                # elif not hasattr(st.session_state,'difficulty'):
                #     st.warning("Please select a difficult level.")
                else:
                    if uploaded_file is not None:
                        st.session_state.context = extract_text_from_pdf(uploaded_file)
                    content_words = nltk.word_tokenize(st.session_state.context)
                    st.session_state.context = " ".join(content_words[:400])
                    prompt = f"Generate {number} Quizzes which require logical reasoning in a list of JSON format from the given context.\nCONTEXT: {st.session_state.context}"
                    st.session_state.quiz_json = {}
                    correctly_generated = False
                    #while not correctly_generated:
                        #try:
                    st.session_state.quiz_json['questions'] = get_response_from_api(prompt)
                            #correctly_generated = True
                        #except:
                           # correctly_generated = False
                    #st.write(st.session_state.quiz_json)
                    #st.session_state.quiz_json['questions'] = st.session_state.quiz_json['questions'][:number]
                    st.session_state.quiz_json['context'] = st.session_state.context
                    # st.write(st.session_state.quiz_json)
                    st.session_state.quiz_generated = True   
                    st.rerun() 
    else:
        # Check if the teacher is currently editing a question
        if st.session_state.get('is_editing', False):
            edit_quiz_question()
        else:
            preview_quiz()
            finalize_quiz()

def option_selection(option):
    st.session_state.selected_option = option
    st.session_state.answer_submitted = True

def display_option_button(option, key):
    if st.button(option, key=key, use_container_width=True):
        option_selection(option)
        # st.rerun()

# Function to check if a topic exists in the data
# Function to check if a topic exists and return its index
def find_topic_index(data, target_topic):
    for index, entry in enumerate(data):
        if entry["topic"] == target_topic:
            return index
    return -1  # Return -1 if the topic is not found

def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    

    if not st.session_state.logged_in:
        st.title('Quiz App')
        st.header("Login")
        st.session_state.username = st.text_input('Username')
        st.session_state.password = st.text_input("Password", type="password")

        if st.button("Login"):
            name, role = check_login(st.session_state.username, st.session_state.password)
            if role:
                st.session_state.logged_in = True
                st.session_state.name = name
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Invalid username or password")
    else:
            # Initialize state variables
        if "quiz_started" not in st.session_state:
            st.session_state.quiz_started = False
        if "quiz_completed" not in st.session_state:
            st.session_state.quiz_completed = False
        if "quiz_data" not in st.session_state:
            st.session_state.quiz_data = []
        if "current_index" not in st.session_state:
            st.session_state.current_index = 0
        if "score" not in st.session_state:
            st.session_state.score = 0
        if "selected_option" not in st.session_state:
            st.session_state.selected_option = None
        if "answer_submitted" not in st.session_state:
            st.session_state.answer_submitted = False
            

        if not st.session_state.quiz_started:
            st.sidebar.success(f"Welcome, {st.session_state.name}")
            if st.sidebar.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.name = None
                st.session_state.role = None
                st.rerun()
            
            if st.session_state.role == "teacher":
                teacher_dashboard()
            elif st.session_state.role == "student":
                st.title("Student's Dashboard - Attempt a Quiz")
                if os.path.exists(f"student_{st.session_state.username}.json"):
                    with open(f"student_{st.session_state.username}.json", "r") as f:
                        st.session_state.student_record = json.load(f)
                else:
                    st.session_state.student_record = None
                if not st.session_state.quiz_started:
                    # Load quizzes created by teachers
                    quizzes = [f for f in os.listdir() if f.startswith("quiz_") and f.endswith(".json")]

                    if len(quizzes)>0:
                        topics = []
                        all_quizzes_data = []
                        attempts = []
                        scores = []
                        num_columns = 1
                        num_rows = (len(quizzes) + num_columns  - 1 ) // num_columns

                        for row_idx in range(num_rows):

                            columns = st.columns(num_columns)

                            for col_idx in range(num_columns):
                                item_idx = row_idx * num_columns + col_idx
                                if item_idx<len(quizzes):
                                    with open(quizzes[item_idx], "r") as f:
                                        quiz_data = json.load(f)
                                        topics.append(quiz_data['topic'])
                                        all_quizzes_data.append(quiz_data)
                                        if st.session_state.student_record:
                                            if quizzes[item_idx] in st.session_state.student_record.keys():
                                                # st.write(len(st.session_state.student_record[quizzes[item_idx]]))
                                                attempts.append(len(st.session_state.student_record[quizzes[item_idx]]))
                                                scores.append([quiz_item['score'] for quiz_item in st.session_state.student_record[quizzes[item_idx]]])
                                                height=300
                                            else:
                                                attempts.append(0)
                                                scores.append([])
                                                height=225
                                        else:
                                            attempts.append(0)
                                            scores.append([])
                                            height = 225
                                    with columns[col_idx].container(height=height):
                                        # with st.expander(f"{item_idx+1}. {quiz_data['topic']}"):
                                    

                                        st.subheader(f"{item_idx+1}. {quiz_data['topic']}")
                                        col1, col2 = st.columns([2,1])
                                        with col1:
                                            st.write(f"Attempts: {attempts[item_idx]}")
                                        with col2:
                                            if len(scores[item_idx]) > 0:
                                                st.write(f"**Score: {max(scores[item_idx])}/{len(quiz_data['questions'])}**")
                                            else:
                                                st.write(f"**Score: 0**")
                                        
                                        col1, col2 = st.columns([2,1])
                                        with col1:
                                            st.write(f"**Questions: {len(quiz_data['questions'])}**")
                                        with col2:
                                            st.write(f"Created by: {users[quizzes[item_idx].split('_')[1]]['name']}")
                                        if len(scores[item_idx]) > 0:
                                            _,col,_ = st.columns([1,1,1])
                                            with col:
                                                with st.expander("Score history"):
                                                    score_attempt = {
                                                        'attempts':range(1,len(scores[item_idx])+1),
                                                        'scores':scores[item_idx]
                                                    }
                                                    st.bar_chart(score_attempt,x="attempts",y="scores", x_label="Attempt", y_label="Score", height=150, width=500)
                                        if st.button("Start Quiz", key = item_idx, use_container_width=True):
                                            # Initialize quiz state
                                            st.session_state.selected_quiz = quizzes[item_idx]
                                            st.session_state.quiz_data = all_quizzes_data[item_idx]['questions']
                                            st.session_state.content = all_quizzes_data[item_idx]['context']
                                            st.session_state.quiz_started = True
                                            st.session_state.current_index = 0
                                            st.session_state.score = 0
                                            st.session_state.selected_option = None
                                            st.session_state.answer_submitted = False
                                            st.session_state.user_answer = [None] * len(st.session_state.quiz_data)
                                            st.session_state.quiz_completed = False
                                            st.session_state.start_time = time.time()
                                            st.session_state.time_elapsed = 0
                                            st.session_state.attempts = attempts[item_idx]
                                            st.rerun()
                                        # st.write("___")
                    else:
                        st.warning("No quizzes available.")
        

        elif not st.session_state.quiz_completed:
            quiz_data = st.session_state.quiz_data
            # st.write(quiz_data)
            # st.write(st.session_state.current_index)
            question_item = quiz_data[st.session_state.current_index]
            # index = find_topic_index(results, st.session_state.quiz_title)
            # if index != -1:
            #     st.session_state.attempts = len(results[index]['attempts'])
            # else:
            #     st.session_state.attempts = 0
            minutes, second = divmod(time.time() - st.session_state.start_time, 60)
            # st.write(st.session_state.user_answer)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader(f"Question {st.session_state.current_index + 1}/{len(quiz_data)}")
            with col2:
                st.subheader(f"Attempt: {st.session_state.attempts+1}")
            with col3:
                st.subheader(f"Time Elapsed: {int(minutes)}:{int(second)}")
            st.title(f"{question_item['question']}")
            st.markdown("""___""")

            distractors = question_item['distractors']
            correct_answer = question_item['correct_option']
            options = distractors + [correct_answer]
            random.seed(st.session_state.attempts+st.session_state.current_index)
            random.shuffle(options)
            for i, option in enumerate(options):
                display_option_button(option, key=i)

            if st.session_state.current_index == len(quiz_data) - 1: # last question
                col1,_, col2 = st.columns(3)
                with col1:
                    if st.button('Previous Question', use_container_width=True):
                        st.session_state.current_index -= 1
                        st.session_state.selected_option = None
                        st.session_state.answer_submitted = False
                        st.rerun()
                with col2:
                    if st.button('Submit Quiz', use_container_width=True):
                        if st.session_state.selected_option is not None:
                            st.session_state.user_answer[st.session_state.current_index] = st.session_state.selected_option
                            st.session_state.quiz_completed = True
                            st.rerun()
            elif st.session_state.current_index == 0: # first question
                col1, _ , col2 = st.columns(3)
                with col2:
                    if st.button('Next Question', use_container_width=True):
                        if st.session_state.selected_option is not None:
                            st.session_state.user_answer[st.session_state.current_index] = st.session_state.selected_option
                            st.session_state.current_index += 1
                            st.session_state.selected_option = None
                            st.session_state.answer_submitted = False
                            st.rerun()
            else:
                col1,_, col2 = st.columns(3)
                with col1:
                    if st.button('Previous Question', use_container_width=True):
                        st.session_state.current_index -= 1
                        st.session_state.selected_option = None
                        st.session_state.answer_submitted = False
                        st.rerun()
                with col2:
                    if st.button('Next Question', use_container_width=True):
                        if st.session_state.selected_option is not None:
                            st.session_state.user_answer[st.session_state.current_index] = st.session_state.selected_option
                            st.session_state.current_index += 1
                            st.session_state.selected_option = None
                            st.session_state.answer_submitted = False
                            st.rerun()
        else:
            st.success("Quiz completed!")
                    # st.write(results)
            if st.session_state.time_elapsed == 0:
                st.session_state.time_elapsed = time.time() - st.session_state.start_time
            quiz_data = st.session_state.quiz_data
            if "solution_viewed" not in st.session_state:
                st.session_state.solution_viewed = False
            st.session_state.score = 0
            for key, value in enumerate(st.session_state.user_answer):
                correct_answer = quiz_data[key]['correct_option']
                quiz_data[key]['selected_option'] = value
                # st.subheader(f"Question {key + 1}")
                # st.subheader(f"{st.session_state.quiz_data[key]['question']}")
                # st.write(f"Correct answer: {correct_answer}")
                if value == correct_answer:
                    # st.success(f"Your answer: {value}")
                    st.session_state.score += 1    
            # st.write(f"Score: {st.session_state.score}/{len(quiz_data)}")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Score: {st.session_state.score} / {len(st.session_state.quiz_data)}")
            with col2:
                minutes, second = divmod(int(st.session_state.time_elapsed), 60)
                st.subheader(f"Time Elapsed: {int(minutes)}:{int(second)}")
            feedback = generate_general_feedback(len(st.session_state.quiz_data),st.session_state.score,(st.session_state.time_elapsed))
            st.markdown("### General Feedback")
            st.write(feedback)
            with st.expander("Solutions"):
                for index, data in enumerate(quiz_data):
                    #right_option_index = data['distractors'].index(data['correct_option'])
                    st.subheader(f"Question {index + 1}")
                    st.subheader(f"Question: {data['question']}")
                    st.success(f"Correct answer: {data['correct_option']}")
                    if data['selected_option'] == data['correct_option']:
                        st.success(f"Your answer: {data['selected_option']}")
                    else:
                        st.error(f"Your answer: {data['selected_option']}")
                        # st.write(data)
                        data['explanation_for_wrong_answer'] = generate_explanation(st.session_state.context, data)
                        st.write(f"Explanation: {data['explanation_for_wrong_answer']}")
            # st.write(data)
            this_attempt_data = {
                "selected_option": [data_item['selected_option'] for data_item in quiz_data],
                "score": st.session_state.score,
                "time_elapsed": st.session_state.time_elapsed
                # "attempt_number": st.session_state.attempts + 1
            }
            # st.write(this_attempt_data)
            
            
            # st.write(st.session_state.student_record[st.session_state.selected_quiz])
            col1,_, col2 = st.columns(3)
            with col1:
                if st.button('Restart Quiz', use_container_width=True):
                    if st.session_state.student_record:
                        if st.session_state.selected_quiz in st.session_state.student_record:
                            st.session_state.student_record[st.session_state.selected_quiz].append(this_attempt_data)
                        else:
                            st.session_state.student_record[st.session_state.selected_quiz] = [this_attempt_data]
                    else:
                        st.session_state.student_record = {}
                        st.session_state.student_record[st.session_state.selected_quiz] = [this_attempt_data]
                    with open(f"student_{st.session_state.username}.json", "w") as f:
                        json.dump(st.session_state.student_record, f)
                    st.session_state.quiz_started = True
                    st.session_state.current_index = 0
                    st.session_state.score = 0
                    st.session_state.selected_option = None
                    st.session_state.answer_submitted = False
                    st.session_state.quiz_completed = False
                    st.session_state.solution_viewed = False
                    st.session_state.attempts+=1
                    st.session_state.start_time = time.time()
                    st.session_state.time_elapsed = 0
                    st.rerun()
            with col2:
                if st.button('Home Page', use_container_width=True):
                    if st.session_state.student_record:
                        if st.session_state.selected_quiz in st.session_state.student_record:
                            st.session_state.student_record[st.session_state.selected_quiz].append(this_attempt_data)
                        else:
                            st.session_state.student_record[st.session_state.selected_quiz] = [this_attempt_data]
                    else:
                        st.session_state.student_record = {}
                        st.session_state.student_record[st.session_state.selected_quiz] = [this_attempt_data]
                    with open(f"student_{st.session_state.username}.json", "w") as f:
                        json.dump(st.session_state.student_record, f)
                    st.session_state.quiz_started = False
                    st.session_state.current_index = 0
                    st.session_state.score = 0
                    st.session_state.selected_option = None
                    st.session_state.answer_submitted = False
                    st.session_state.quiz_completed = False
                    st.rerun()
if __name__ == "__main__":
    main()
