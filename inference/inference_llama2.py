# This code was derived and modified from https://github.com/dvlab-research/LongLoRA/blob/main/inference.py

import sys
import math
import torch
import argparse
import transformers
import json
from distutils.util import strtobool
from utils.conversation import get_conv_template
from utils.llama_flash_attn_longlora import replace_llama_attn
from utils.common import extract_text, format_inference_prompt

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default="./data", help='path to the directory of the ELITR transcripts and ELITR-Bench JSON file')
    parser.add_argument('--json_filename', type=str, default="elitr-bench-qa_test2.json", help='name of the ELITR-Bench JSON file')
    parser.add_argument('--base_model', type=str, default="lmsys/vicuna-13b-v1.5-16k", help='name or path of the HuggingFace model used for response generation')
    parser.add_argument('--cache_dir', type=str, default="./cache", help='cache directory path')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=lambda x: bool(strtobool(x)), default=False, help='whether to use Flash Attention')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature hyperparameter')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p hyperparameter')
    parser.add_argument('--max_gen_len', type=int, default=512, help='maximum generation length')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition penalty hyperparameter')
    parser.add_argument('--do_sample', type=lambda x: bool(strtobool(x)), default=False, help='use random sampling instead of greedy decoding')
    parser.add_argument('--conv_format', type=str, default="raw", choices=["raw", "llama-2", "vicuna_v1.1", "longalign"], help='chat template to be used (raw for no chat template)')
    parser.add_argument('--q_marker', type=lambda x: bool(strtobool(x)), default=False, help='add QUESTION and ANSWER markers to prompt')
    parser.add_argument('--mode', type=str, default="st", choices=["st", "mt"], help='use single-turn (st) or multi-turn (mt) mode')
    parser.add_argument('--seed', type=int, default=2023, help='seed used for random sampling')
    args = parser.parse_args()
    return args

def build_generator(
    model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=512, use_cache=True, do_sample=False, repetition_penalty=1.0
):

    def response(context):
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
        print(inputs['input_ids'].size()[1])

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty
        )

        out = tokenizer.decode(output[0], skip_special_tokens=False)
        out = out.split(context)[1].rstrip("</s>").strip()

        return out

    return response

def main(args):
    model_name = args.base_model.split("/")[-1]
    if args.flash_attn:
        replace_llama_attn()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and args.context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(args.context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
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
                conv = get_conv_template(args.conv_format)
                conv.append_message(conv.roles[0], initial_message + "\n\n" + formatted_question)
            else:
                conv.append_message(conv.roles[0], formatted_question)

            # Assistant turn
            conv.append_message(conv.roles[1], None)
            context = conv.get_prompt()
            output = respond(context)
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
    with open(new_json_file_path, 'w') as f:
        json.dump(elitrbench_dict, f, indent=2)

if __name__ == "__main__":
    args = parse_config()
    print(vars(args), flush=True)

    transformers.set_seed(args.seed)
    main(args)
