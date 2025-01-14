def extract_text(file_path):
    file_str = ""
    with open(file_path) as f:
        for line in f.readlines():
            file_str += line
    return file_str

def format_inference_prompt(transcript):
    prompt = (f"The following is the transcript of a meeting with multiple participants, "
              f"where utterances start with the speaker's anonymized name (for instance (PERSON4)) and may span over several lines.\n\n"
              f"{transcript}\n\n"
              f"As a professional conversational assistant, your task is to answer questions about the meeting "
              f"by making inferences from the provided transcript.\n\n")

    return prompt

def format_gpt_eval_prompt(question, response, reference):
    prompt = (f"### Task description:\n"
              f"You are provided below with a question, a response to evaluate, a reference answer that gets the maximum score of 10, and a score rubric representing evaluation criteria.\n"
              f"1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.\n"
              f"2. After writing a feedback, write a score that is an integer between 1 and 10. You should refer to the score rubric.\n"
              f"3. The output format should first include the feedback and then indicate the integer score in \\boxed{{}}.\n"
              f"4. Please do not generate any other opening, closing, and explanations.\n\n"
              f"### Question:\n"
              f"{question}\n\n"
              f"### Response to evaluate:\n"
              f"{response}\n\n"
              f"### Reference answer (score 10):\n"
              f"{reference}\n\n"
              f"### Score rubric:\n"
              f"[Does the response to evaluate correctly address the given question based on the elements provided by the reference answer? "
              f"The response should include the elements of the reference answer and should also avoid adding unnecessary elements or being too verbose.]\n"
              f"Score 1: The response to evaluate is incorrect and misses all the elements of the reference answer.\n"
              f"Score 2: The response to evaluate indicates insufficient knowledge to answer the question even though the reference answer states otherwise.\n"
              f"Score 3-4: The response to evaluate contains some elements vaguely related to the reference answer.\n"
              f"Score 5-6: The response to evaluate is partially correct and/or covers only a part of the reference answer.\n"
              f"Score 7-8: The response to evaluate contains most of the reference answer but delivers it in an indirect and/or overly verbose way.\n"
              f"Score 9: The response to evaluate includes the reference answer but it is more verbose and adds unnecessary elements.\n"
              f"Score 10: The response to evaluate is essentially equivalent to the reference answer.\n\n"
              f"### Feedback:\n")

    return prompt

def format_prometheus_eval_prompt(question, response, reference):
    prompt = (f"### Task description:\n"
              f"You are provided below with a question, a response to evaluate, a reference answer that gets the maximum score of 5, and a score rubric representing evaluation criteria.\n"
              f"1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.\n"
              f"2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.\n"
              f"3. The output format should look as follows: \"Feedback: (write the quality assessment feedback) [RESULT] (an integer number between 1 and 5)\".\n"
              f"4. Please do not generate any other opening, closing, and explanations.\n\n"
              f"### Question:\n"
              f"{question}\n\n"
              f"### Response to evaluate:\n"
              f"{response}\n\n"
              f"### Reference answer (score 5):\n"
              f"{reference}\n\n"
              f"### Score rubric:\n"
              f"[Does the response to evaluate correctly address the given question based on the elements provided by the reference answer? "
              f"The response should include the elements of the reference answer and should also avoid adding unnecessary elements or being too verbose.]\n"
              f"Score 1: The response to evaluate is incorrect and misses all the elements of the reference answer.\n"
              f"Score 2: The response to evaluate contains some elements vaguely related to the reference answer.\n"
              f"Score 3: The response to evaluate is partially correct and/or covers only a part of the reference answer.\n"
              f"Score 4: The response to evaluate contains most of the reference answer but delivers it in an indirect and/or overly verbose way.\n"
              f"Score 5: The response to evaluate is essentially equivalent to the reference answer.\n"
              f"### Feedback:\n")

    return prompt