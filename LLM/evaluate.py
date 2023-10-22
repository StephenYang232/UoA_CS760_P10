import os
import sys

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

from sklearn.metrics import roc_auc_score, accuracy_score
import json

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

torch.set_num_threads(1)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def main(
    load_8bit: bool = False,
    load_4bit: bool = False,
    use_flash_attention_2: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    train_data_name: str = "",
    test_data_name: str = "",
    result_data: str = "",
    batch_size = 32,
    token: str = ""
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    if not os.path.exists(result_data):
        with open(result_data, "x") as f:
            f.write("train_data_name,test_data_name,model_name,accuracy,auc\n")

    model_name = lora_weights.split('/')[-1]
    test_data_path = os.path.join(test_data_name, "test.jsonl")

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        load_8bit = False if load_4bit else load_8bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            use_flash_attention_2=use_flash_attention_2,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token or None
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )


    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not (load_8bit or load_4bit):
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = torch.tensor(scores[:,[8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        # logits = torch.tensor(scores[:,[8241, 3782]] + scores[:, [3869, 1939]], dtype=torch.float32).softmax(dim=-1)
        return output, logits[:, 0].tolist()

    # testing code for readme
    from tqdm import tqdm

    with open(test_data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]

    instructions = [_['instruction'] for _ in test_data]
    inputs = [_['input'] for _ in test_data]
    outputs = []
    logits = []
    def batches(list):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    for i, batch in tqdm(enumerate(zip(batches(instructions), batches(inputs)))):
        instructions, inputs = batch
        output, logit = evaluate(instructions, inputs)
        outputs = outputs + output
        logits = logits + logit

    gold = [int(_['output'] == 'Yes.') for _ in test_data]
    pred = [int(_ == 'Yes.') if _ == "Yes." or _ == "No." else -1 for _ in outputs]

    with open(result_data, "a") as f:
        f.write(f"{train_data_name},{test_data_name},{model_name},{roc_auc_score(gold, logits)},{accuracy_score(gold, pred)}\n")

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)