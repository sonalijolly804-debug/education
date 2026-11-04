import argparse
import json
import re
from pathlib import Path
from common import ROOT, instruction, read_records, target, user_message, validate_record


def main():
    parser = argparse.ArgumentParser(description='Generate slide-aligned lecture narration.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--model', default='unsloth/Qwen3.5-4B')
    parser.add_argument('--method', choices=['zero_shot', 'few_shot', 'cot', 'train'], default='zero_shot')
    parser.add_argument('--unguided', action='store_true')
    parser.add_argument('--examples', type=Path)
    parser.add_argument('--shots', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=30000)
    parser.add_argument('--max-seq-length', type=int, default=30000)
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = read_records(args.data)
    if args.output.exists():
        parser.error('Output already exists; choose a new output path.')
    examples = read_records(args.examples) if args.examples else []
    if args.method == 'few_shot' and (not examples or args.shots < 1):
        parser.error('Few-shot generation requires --examples and a positive --shots.')
    keys = {(r['course_id'], r['lecture_number']) for r in records}
    if any((r['course_id'], r['lecture_number']) in keys for r in examples):
        parser.error('Few-shot examples must come from different lectures than the inputs.')
    for record in records:
        validate_record(record, args.image_root, require_annotation=False)
    for example in examples:
        validate_record(example, args.image_root, require_annotation=not args.unguided)
    from unsloth import FastVisionModel
    import torch
    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=args.load_in_4bit, max_seq_length=args.max_seq_length)
    FastVisionModel.for_inference(model)
    prompt = None if args.method == 'train' else instruction(args.method, not args.unguided)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for record in records:
            messages = []
            if args.method == 'few_shot':
                candidates = [e for e in examples if e['category'] == record['category']]
                if len(candidates) < args.shots:
                    raise ValueError(f"Not enough demonstrations for {record['category']}")
                for example in candidates[:args.shots]:
                    messages.extend([user_message(example, prompt, args.image_root),
                        {'role': 'assistant', 'content': [{'type': 'text',
                         'text': '<answer>' + target(example, not args.unguided) + '</answer>'}]}])
            messages.append(user_message(record, prompt, args.image_root))
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors='pt', return_dict=True).to(model.device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            raw = processor.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            answer = re.search(r'<answer>(.*?)</answer>', raw, re.S)
            text = answer[1].strip() if answer else re.sub(r'<think>.*?</think>', '', raw, flags=re.S).strip()
            handle.write(json.dumps({'id': record['id'], 'generated_transcript': text}, ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    main()
