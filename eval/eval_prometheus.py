import sys
import torch
import argparse
import transformers
import json
from distutils.util import strtobool
import re
from utils.conversation import get_conv_template
from utils.common import format_prometheus_eval_prompt

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default="./data", help='path to the directory of the ELITR transcripts and ELITR-Bench JSON file')
    parser.add_argument('--json_filename', type=str, default="elitr-bench-qa_test2.json", help='name of the ELITR-Bench JSON file with generated responses')
    parser.add_argument('--base_model', type=str, default="prometheus-eval/prometheus-13b-v1.0", help='name of the Prometheus model used for response evaluation')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature hyperparameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p hyperparameter')
    parser.add_argument('--max_gen_len', type=int, default=1024, help='maximum generation length')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition penalty hyperparameter')
    parser.add_argument('--do_sample', type=lambda x: bool(strtobool(x)), default=True, help='')
    parser.add_argument('--seed', type=int, default=2023, help='seed used for random sampling')
    args = parser.parse_args()
    return args

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=1024, use_cache=True, do_sample=True, repetition_penalty=1.0
):

    def response(context):
        inputs = tokenizer(context, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty
        )

        out = tokenizer.decode(output[0], skip_special_tokens=True)
        out = out.split(context.lstrip("<s>"))[1].strip()

        return out

    return response

def main(args):
    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=True, do_sample=args.do_sample,
                              repetition_penalty=args.repetition_penalty)

    # Evaluate responses
    json_file_path = args.data_path + "/" + args.json_filename

    with open(json_file_path, 'r') as f:
        elitrbench_dict = json.loads(f.read())

    model_name = args.base_model.split("/")[-1]
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
                prompt = format_prometheus_eval_prompt(formatted_question, generated_response, groundtruth_answer)

                # Convert the prompt the OpenAI format
                conv = get_conv_template("llama-2")
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                context = conv.get_prompt()

                # Generate the feedback and parse the score
                feedback = respond(context)
                print("### prompt ###", flush=True)
                print(prompt, flush=True)
                print("### feedback ###", flush=True)
                print(feedback, flush=True)
                score = re.search(r'\[RESULT\] (\d+)', feedback)
                if score is not None:
                    score = score.group(1)
                else: # Manual check required
                    print("[RESULT] marker not found, using placeholder score 0", flush=True)
                    score = "0"
                response_dict[model_name + "-eval_score"] = str(2 * int(score)) # Transform to a 1-10 scale

    str_split = json_file_path.split(".")
    new_json_file_path = ".".join(str_split[:-1]) + "_eval_" + model_name + "." + str_split[-1]
    with open(new_json_file_path, 'w') as f:
        json.dump(elitrbench_dict, f, indent=2)

if __name__ == "__main__":
    args = parse_config()
    print(vars(args), flush=True)

    transformers.set_seed(args.seed)
    main(args)
