import argparse
import json
import os
import torch
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.datasets.radvqa import RADVQADataset
from src.highlighter_modules.guidance import ProbCFGLogitsProcessor


# suggested deepspeed config
DS_CONFIG_DICT = {
    'zero_optimization': {
        'stage': 2,
        'allgather_partitions': True,
        'allgather_bucket_size': 5e8,
        'overlap_comm': True,
        'reduce_scatter': True,
        'reduce_bucket_size': 5e8,
        'contiguous_gradients': True,
        'round_robin_gradients': True,
    },
    'fp16': {
        'enabled': 'auto',
        'loss_scale': 0,
        'loss_scale_window': 1000,
        'initial_scale_power': 16,
        'hysteresis': 2,
        'min_loss_scale': 1,
    },
    'bf16': {'enabled': 'auto'},
    'train_micro_batch_size_per_gpu': 'auto',
    'train_batch_size': 'auto',
    'gradient_accumulation_steps': 'auto',
    'gradient_clipping': 'auto',
}


def create_dataset(args):
    output_file_test = args.input_json
    img_root = args.img_root
    eval_dataset = RADVQADataset(annotation_file=output_file_test, vis_root=img_root)
    return eval_dataset



class NoGradHook:
    def __init__(self):
        self.prev_enabled = True

    def maybe_enable_grad_hook(self, *_):
        torch.set_grad_enabled(self.prev_enabled)

    def disable_grad_hook(self, *_):
        self.prev_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)


def freeze_vision_model(model):
    vision_no_grad_hook = NoGradHook()
    vision_module = model.model.vision_embed_tokens
    vision_module.register_forward_pre_hook(vision_no_grad_hook.disable_grad_hook)
    vision_module.register_forward_hook(vision_no_grad_hook.maybe_enable_grad_hook)
    for p in vision_module.parameters():
        p.requires_grad_(False)


def create_model(model_name_or_path, use_flash_attention=False, use_qlora=False):
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention else torch.float16,
        )
        if use_qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config=bnb_config,
    )

    return model


@torch.no_grad()
def evaluate(model, processor, eval_dataset, args, save_path=None, disable_tqdm=False):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    model.eval()
    answers_unique = []
    generated_texts_unique = []

    eval_dataset_shard = eval_dataset

    for i in tqdm(range(len(eval_dataset)), disable=(rank != 0) or disable_tqdm):
        # Phi-3-V currently only supports batch_size == 1
        example = eval_dataset_shard[i]
        answers_unique.append(example['answer'])
        answers_unique.append(example['answer'])
        image = example['image']
        question = example['question']
        caption = example["top_k_captions"][0]
        highlights = example['highlights']
        prompt_message = {
            'role': 'user',
            'content': f'{caption} <|image_1|>\n{question}',
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        qs_highlighted_parts = highlights

        inputs = processor(prompt, [image], return_tensors='pt', qs_highlighted_parts=qs_highlighted_parts).to(f'cuda:{local_rank}')
        hl_mask_ = inputs['highlight_attention_mask']
        hl_mask_[hl_mask_ == 1] = args.perturb_weight
        hl_mask_[hl_mask_ == 0] = args.attn
        cfg_batched_input = inputs['input_ids'].repeat(2, 1)
        pixel_values = inputs['pixel_values'].repeat(2, 1, 1, 1, 1)
        image_sizes = inputs['image_sizes'].repeat(2, 1)

        del inputs['highlight_attention_mask']

        generated_outputs = model.generate(
            input_ids=cfg_batched_input,
            pixel_values=pixel_values,
            attention_mask=torch.cat([inputs['attention_mask'], hl_mask_], dim=0),
            image_sizes=image_sizes,
            eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=64,
            num_beams=args.num_beams,
            logits_processor=[ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=True)],
            output_scores=True,
            return_dict_in_generate=True
        )


        batch_index = 1
        prediction = processor.batch_decode(
            generated_outputs.sequences[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        prediction = prediction[0].strip().strip('.')

        print('Question:', example['question'], 'GT:',example['answer'])
        print('Prediction:', prediction)
        token_probs = []
        generated_texts = []
        for i, scores in enumerate(generated_outputs.scores):
            probs = torch.softmax(scores, dim=-1)
            generated_token_id = generated_outputs.sequences[batch_index, inputs['input_ids'].size(1) + len(token_probs)]
            token_prob = probs[batch_index, generated_token_id].item()
            token_probs.append(token_prob)

        # Print the decoded tokens and their probabilities
        print("Generated text and token probabilities:")
        for idx, prob in enumerate(token_probs):
            token = processor.decode(generated_outputs.sequences[batch_index, inputs['input_ids'].size(1) + idx])
            print(f"{token} - Probability: {prob}")
            generated_texts.append(token)

        update_information = {
            'question': example['question'],
            'answer': example['answer'],
            'prediction': prediction,
            'token_probs': token_probs,
            'token_preds': generated_texts
        }
        generated_texts_unique.append(update_information)



    if rank == 0:
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(generated_texts_unique, f, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        # default='./Phi-3-vision-128k-instruct',
        default='./Phi-3.5-vision-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLora')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--save_path', type=str, help='Save json path')
    parser.add_argument('--input_json', type=str, help='Question and Answer json path')
    parser.add_argument('--img_root', type=str, help='Image Folder')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument(
        '--lora_alpha_ratio', type=float, default=2, help='LoRA alpha to rank ratio'
    )
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--freeze_vision_model', action='store_true', help='Freeze vision model')

    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--attn", type=float, default=3.0)
    parser.add_argument("--perturb_weight", type=float, default=0.01)

    args = parser.parse_args()
    args.attention_weight = args.attn

    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'
    if args.use_qlora:
        args.use_lora = True

    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=None,
        split_batches=False,
        even_batches=True,
        use_seedable_sampler=True
    )
    accelerator = Accelerator(dataloader_config)

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
            use_qlora=args.use_qlora,
        )

    eval_dataset = create_dataset(args)

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert args.batch_size % num_gpus == 0, 'Batch size must be divisible by the number of GPUs'


    # eval after fine-tuning
    if args.use_lora:
        # first try to clear GPU memory
        del model
        __import__('gc').collect()
        torch.cuda.empty_cache()

        # reload the model for inference
        # this part also serves as an example of how to load a trained model
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            # Phi-3-V is originally trained in bf16 + flash attn
            # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
            torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
        )
        model.load_adapter(args.output_dir)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(f'cuda:{local_rank}')
    evaluate(
        model,
        processor,
        eval_dataset,
        args,
        save_path=args.save_path,
        disable_tqdm=not args.tqdm,
    )



if __name__ == '__main__':
    main()
