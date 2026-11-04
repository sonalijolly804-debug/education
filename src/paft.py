import argparse
from pathlib import Path
from common import ROOT, read_records, user_message, validate_record


def main():
    parser = argparse.ArgumentParser(description='Pedagogy-aware fine-tuning with joint tag and narration supervision.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--model', default='unsloth/Qwen3.5-4B')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=3)
    parser.add_argument('--gradient-accumulation', type=int, default=6)
    parser.add_argument('--epochs', type=float, default=10)
    parser.add_argument('--max-seq-length', type=int, default=30000)
    parser.add_argument('--load-in-4bit', action='store_true')
    args = parser.parse_args()
    records = read_records(args.data)
    for record in records:
        validate_record(record, args.image_root, require_annotation=True)
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer
    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=args.load_in_4bit,
        use_gradient_checkpointing='unsloth', max_seq_length=args.max_seq_length)
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=16, lora_alpha=16, lora_dropout=0, bias='none',
        random_state=3407, use_rslora=False, loftq_config=None)
    dataset = [{'messages': [user_message(r, root=args.image_root),
               {'role': 'assistant', 'content': [{'type': 'text', 'text': r['annotated_transcript']}]}]}
               for r in records]
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model, tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(args.output), per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            num_train_epochs=args.epochs, warmup_steps=5, learning_rate=2e-4,
            optim='adamw_8bit', weight_decay=0.001, lr_scheduler_type='linear',
            seed=3407, report_to='none', logging_strategy='no',
            remove_unused_columns=False, dataset_text_field='',
            dataset_kwargs={'skip_prepare_dataset': True},
            save_strategy='epoch', save_total_limit=3, eval_strategy='no'))
    trainer.train()
    model.save_pretrained(str(args.output))
    processor.save_pretrained(str(args.output))


if __name__ == '__main__':
    main()
