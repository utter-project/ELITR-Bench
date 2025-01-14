import torch
import argparse
import transformers
import json
from distutils.util import strtobool
from utils.common import extract_text, format_inference_prompt

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default="./data", help='path to the directory of the ELITR transcripts and ELITR-Bench JSON file')
    parser.add_argument('--json_filename', type=str, default="elitr-bench-qa_test2.json", help='name of the ELITR-Bench JSON file')
    parser.add_argument('--base_model', type=str, default="microsoft/Phi-3-small-128k-instruct", help='name or path of the HuggingFace model used for response generation')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature hyperparameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p hyperparameter')
    parser.add_argument('--max_gen_len', type=int, default=512, help='maximum generation length')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition penalty hyperparameter')
    parser.add_argument('--do_sample', type=lambda x: bool(strtobool(x)), default=False, help='use random sampling instead of greedy decoding')
    parser.add_argument('--q_marker', type=lambda x: bool(strtobool(x)), default=False, help='add QUESTION and ANSWER markers to prompt')
    parser.add_argument('--mode', type=str, default="st", choices=["st", "mt"], help='use single-turn (st) or multi-turn (mt) mode')
    parser.add_argument('--seed', type=int, default=2023, help='seed used for random sampling')
    args = parser.parse_args()
    return args

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=512, use_cache=True, do_sample=False, repetition_penalty=1.0
):

    def response(context):
        inputs = tokenizer.apply_chat_template(context, add_generation_prompt=True, return_tensors="pt").to(model.device)

        output = model.generate(
            inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty
        )

        out = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        out = out.rstrip("</s>")

        return out

    return response

def main(args):
    model_name = args.base_model.split("/")[-1]

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    model.eval()
    respond = build_generator(model, tokenizer, temperature=args.temperature, top_p=args.top_p,
                              max_gen_len=args.max_gen_len, use_cache=True, do_sample=args.do_sample,
                              repetition_penalty=args.repetition_penalty)

    # Generate responses
    json_file_path = args.data_path + "/" + args.json_filename

    with open(json_file_path, 'r') as f:
        elitrbench_dict = json.loads(f.read())

    for meeting in elitrbench_dict['meetings']:
        meeting_id = meeting["id"]
        print(meeting_id, flush=True)
        questions = meeting["questions"]

        # Define the initial message containing the basic instructions and the transcript
        transcript_path = args.data_path + "/" + meeting_id + ".txt"
        transcript = extract_text(transcript_path)
        initial_message = format_inference_prompt(transcript)

        for id, question in enumerate(questions):
            print(question["id"], flush=True)

            # User turn
            if args.q_marker:
                formatted_question = "QUESTION: " + question["question"].strip() + " ANSWER:"
            else:
                formatted_question = question["question"].strip()

            if (args.mode == "mt" and id == 0) or args.mode == "st":
                context = []
                context.append({"role": "user", "content": initial_message + "\n\n" + formatted_question})
            else:
                context.append({"role": "user", "content": formatted_question})

            # Assistant turn
            output = respond(context)
            if args.mode == "mt":
                context.append({"role": "assistant", "content": output})

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
    with open(new_json_file_path, 'w') as f:
        json.dump(elitrbench_dict, f, indent=2)

if __name__ == "__main__":
    args = parse_config()
    print(vars(args), flush=True)

    transformers.set_seed(args.seed)
    main(args)
