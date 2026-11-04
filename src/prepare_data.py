import argparse
import ast
import csv
import json
from pathlib import Path
from common import ROOT, normalize, validate_record


def main():
    parser = argparse.ArgumentParser(description='Convert lecture CSV records to JSONL.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-root', type=Path, default=ROOT)
    parser.add_argument('--strip-prefix', default='')
    parser.add_argument('--transcript-column', default='transcript_merged')
    parser.add_argument('--annotation-column', default='xml_train_annotated_transcript_clean')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new output path.')
    csv.field_size_limit(100_000_000)
    records = []
    keys = set()
    with args.input.open(encoding='utf-8', newline='') as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            paths = ast.literal_eval(row['slide_set'])
            if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
                raise ValueError('slide_set must contain a list of image paths.')
            if args.strip_prefix:
                prefix = args.strip_prefix.rstrip('/') + '/'
                if any(not p.startswith(prefix) for p in paths):
                    raise ValueError('An image path does not match --strip-prefix.')
                paths = [p[len(prefix):] for p in paths]
            key = (row['course_id'], row['lecture_number'], tuple(paths))
            if key in keys:
                raise ValueError(f'Duplicate lecture/slide record: {key}')
            keys.add(key)
            record = dict(id=row.get('id') or f'lecture_{i+1:06d}', course_id=row['course_id'],
                          lecture_number=row['lecture_number'], category=row['category'],
                          slide_set=paths, transcript=normalize(row[args.transcript_column]))
            annotation = row.get(args.annotation_column, '').strip()
            if annotation:
                record['annotated_transcript'] = annotation.replace('non-verbal_cue', 'non_verbal_cue')
            validate_record(record, args.image_root, require_annotation=bool(annotation))
            if len(record['transcript'].split('[NEXT_SLIDE]')) != len(paths):
                raise ValueError(f'Slide/transcript count mismatch: {key}')
            records.append(record)
    if not records:
        raise ValueError('The input dataset is empty.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
