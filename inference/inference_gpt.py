import argparse
import json
import os
import openai
from utils.conversation import get_conv_template
from utils.common import extract_text, format_inference_prompt

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default="./data", help='path to the directory of the ELITR transcripts and ELITR-Bench JSON file')
    parser.add_argument('--json_filename', type=str, default="elitr-bench-qa_test2.json", help='name of the ELITR-Bench JSON file')
    parser.add_argument('--base_model', type=str, default="gpt-4-1106-preview", choices=["gpt-3.5-turbo-16k-0613", "gpt-4-1106-preview", "gpt-4o-2024-05-13"], help='name of the OpenAI model used for response generation')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature hyperparameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p hyperparameter')
    parser.add_argument('--max_gen_len', type=int, default=512, help='maximum generation length')
    parser.add_argument('--mode', type=str, default="st", choices=["st", "mt"], help='use single-turn (st) or multi-turn (mt) mode')
    parser.add_argument('--seed', type=int, default=2023, help='seed used for random sampling')
    parser.add_argument('--lang', type=str, default="english", choices=["english", "czech", "en", "cz"], help='language of the questions and expected answers')
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

    # Generate responses
    json_file_path = args.data_path + "/" + args.json_filename

    with open(json_file_path, 'r') as f:
        elitrbench_dict = json.loads(f.read())

    model_name = args.base_model
    for meeting in elitrbench_dict['meetings']:
        meeting_id = meeting["id"]
        print(meeting_id, flush=True)
        questions = meeting["questions"]

        transcript_path = args.data_path + "/" + meeting_id + ".txt"
        transcript = extract_text(transcript_path)
        system_message = format_inference_prompt(transcript, args.lang)
        if args.mode == "mt":
            conv = get_conv_template("chatgpt")
            conv.set_system_message(system_message)

        for question in questions:
            print(question["id"], flush=True)
            if args.mode == "st":
                conv = get_conv_template("chatgpt")
                conv.set_system_message(system_message)

            # User turn
            formatted_question = question["question"].strip()
            conv.append_message(conv.roles[0], formatted_question)
            conv.append_message(conv.roles[1], None)
            context = conv.to_openai_api_messages()

            # Assistant turn
            output = respond({"messages": context})
            if args.mode == "mt":
                conv.update_last_message(output)

            # Update dictionary with the generated answer
            if "generated-responses" not in question.keys():
                question["generated-responses"] = []
            response_dict = {"model": model_name, "generated-response": output}
            question["generated-responses"].append(response_dict)

            print("### question ###", flush=True)
            print(formatted_question, flush=True)
            print("### groundtruth answer ###", flush=True)
            print(question["groundtruth-answer"].strip(), flush=True)
            print("### generated answer ###", flush=True)
            print(output, flush=True)
            print("######", flush=True)

    str_split = json_file_path.split(".")
    new_json_file_path = ".".join(str_split[:-1]) + "_" + args.mode + "_s" + str(args.seed) + "_" + model_name + "." + str_split[-1]
    with open(new_json_file_path, 'w', encoding="utf-8") as f:
        json.dump(elitrbench_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    args = parse_config()
    print(vars(args), flush=True)

    main(args)
