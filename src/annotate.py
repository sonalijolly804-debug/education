import argparse
import json
import os
from pathlib import Path
from common import BOUNDARY, ROOT, TAGS, normalize, read_records, validate_record


def render_spans(sentences, spans):
    cursor = 1
    output = []
    for span in spans:
        start, end = span['start_index'], span['end_index']
        label = span['label'].strip().lower().replace(' ', '_').replace('-', '_')
        if label not in TAGS or type(start) is not int or type(end) is not int:
            raise ValueError('Invalid label or sentence index.')
        if start != cursor or not start <= end <= len(sentences):
            raise ValueError('Spans must cover each sentence exactly once and in order.')
        output.append(f"<{label}> {' '.join(sentences[start-1:end])} </{label}>")
        cursor = end + 1
    if cursor != len(sentences) + 1:
        raise ValueError('Annotation does not cover the complete transcript.')
    return ' '.join(output)


def main():
    parser = argparse.ArgumentParser(description='Annotate transcript spans with pedagogical labels.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new output path.')
    if not os.environ.get('GEMINI_API_KEY'):
        parser.error('Set GEMINI_API_KEY in your environment.')
    from google import genai
    from google.genai import types
    from nltk.tokenize import sent_tokenize
    records = read_records(args.data)
    for record in records:
        validate_record(record, args.image_root, require_annotation=False)
    client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    prompt = (ROOT / 'prompts/annotation.txt').read_text(encoding='utf-8')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for record in records:
            blocks = []
            for block in record['transcript'].split(BOUNDARY):
                sentences = sent_tokenize(normalize(block))
                if not sentences:
                    raise ValueError(f"Empty transcript block in {record['id']}")
                indexed = '\n'.join(f's{i}: {s}' for i, s in enumerate(sentences, 1))
                response = client.models.generate_content(
                    model=args.model, contents=prompt.replace('{transcript}', indexed),
                    config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0))
                blocks.append(render_spans(sentences, json.loads(response.text)))
            record = dict(record, annotated_transcript=f' {BOUNDARY} '.join(blocks),
                          annotation_source=args.model)
            validate_record(record, args.image_root)
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            handle.flush()


if __name__ == '__main__':
    main()
