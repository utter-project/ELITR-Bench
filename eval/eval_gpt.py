import argparse
import json
import os
import openai
import re
from utils.conversation import get_conv_template
from utils.common import format_gpt_eval_prompt

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default="./data", help='path to the directory of the ELITR transcripts and ELITR-Bench JSON file')
    parser.add_argument('--json_filename', type=str, default="elitr-bench-qa_test2.json", help='name of the ELITR-Bench JSON file with generated responses')
    parser.add_argument('--base_model', type=str, default="gpt-4-0613", help='name of the OpenAI model used for response evaluation')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature hyperparameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p hyperparameter')
    parser.add_argument('--max_gen_len', type=int, default=1024, help='maximum generation length')
    parser.add_argument('--seed', type=int, default=2023, help='seed used for random sampling')
    args = parser.parse_args()
    return args

def main(args):
    # OpenAI client
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])

    # Define the text completion function
    default = {
        "model": args.base_model,
        "max_tokens": args.max_gen_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed
    }
    respond = lambda us: client.chat.completions.create(**{**default, **us}).choices[0].message.content

    # Evaluate responses
    json_file_path = args.data_path + "/" + args.json_filename

    with open(json_file_path, 'r') as f:
        elitrbench_dict = json.loads(f.read())

    model_name = args.base_model
    for meeting in elitrbench_dict['meetings']:
        meeting_id = meeting["id"]
        print(meeting_id, flush=True)
        questions = meeting["questions"]

        for question in questions:
            print(question["id"], flush=True)
            formatted_question = question["question"].strip()
            groundtruth_answer = question["groundtruth-answer"].strip()

            for response_dict in question["generated-responses"]:
                # Define the prompt
                generated_response = response_dict["generated-response"].strip()
                prompt = format_gpt_eval_prompt(formatted_question, generated_response, groundtruth_answer)

                # Convert the prompt the OpenAI format
                conv = get_conv_template("chatgpt")
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                context = conv.to_openai_api_messages()

                # Generate the feedback and parse the numeric score
                feedback = respond({"messages": context})
                print("### prompt ###", flush=True)
                print(prompt, flush=True)
                print("### feedback ###", flush=True)
                print(feedback, flush=True)
                score = re.search(r'\\boxed{(\d+)}', feedback)
                if score is not None:
                    score = score.group(1)
                else: # Manual check required
                    print("Boxed score not found, using placeholder score 0", flush=True)
                    score = "0"
                response_dict[model_name + "-eval_score"] = score

    str_split = json_file_path.split(".")
    new_json_file_path = ".".join(str_split[:-1]) + "_eval_" + model_name + "." + str_split[-1]
    with open(new_json_file_path, 'w') as f:
        json.dump(elitrbench_dict, f, indent=2)

if __name__ == "__main__":
    args = parse_config()
    print(vars(args), flush=True)

    main(args)
