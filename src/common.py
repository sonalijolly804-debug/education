import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAGS = ('organization', 'explanation', 'illustration', 'question',
        'contrast', 'conclusion', 'non_verbal_cue')
BOUNDARY = '[NEXT_SLIDE]'
TAG_PATTERN = re.compile(r'</?(?:' + '|'.join(TAGS) + r')>')
SPAN_PATTERN = re.compile(r'<(' + '|'.join(TAGS) + r')>(.*?)</\1>', re.S)


def normalize(text):
    return ' '.join(text.split())


def plain_text(text):
    return normalize(TAG_PATTERN.sub('', text))


def read_records(path):
    path = Path(path)
    files = sorted(path.glob('*.jsonl')) if path.is_dir() else [path]
    records = []
    for file in files:
        with file.open(encoding='utf-8') as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    if not records:
        raise ValueError('The input dataset is empty.')
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Record IDs must be unique.')
    return records


def validate_record(record, root=ROOT, require_annotation=True):
    paths = record['slide_set']
    if not isinstance(paths, list) or not paths:
        raise ValueError('slide_set must be a nonempty list.')
    for path in paths:
        if not (Path(root) / path).is_file():
            raise FileNotFoundError(path)
    if not require_annotation:
        return
    blocks = record['annotated_transcript'].split(BOUNDARY)
    if len(blocks) != len(paths):
        raise ValueError(f"Slide count mismatch in {record['id']}")
    for block in blocks:
        matches = list(SPAN_PATTERN.finditer(block))
        if not matches or SPAN_PATTERN.sub('', block).strip():
            raise ValueError(f"Invalid pedagogical markup in {record['id']}")
        if any(not m[2].strip() or TAG_PATTERN.search(m[2]) for m in matches):
            raise ValueError(f"Empty or nested span in {record['id']}")
    if plain_text(record['annotated_transcript']) != normalize(record['transcript']):
        raise ValueError(f"Annotation changes source text in {record['id']}")


def instruction(method, guided):
    name = f"{method}_{'guided' if guided else 'unguided'}.txt"
    return (ROOT / 'prompts' / name).read_text(encoding='utf-8')


def image_content(record, root=ROOT, label_slides=True):
    from PIL import Image
    content = []
    for i, path in enumerate(record['slide_set'], 1):
        with Image.open(Path(root) / path) as image:
            rgb = image.convert('RGB')
        if label_slides:
            content.append({'type': 'text', 'text': f'Slide {i}:'})
        content.append({'type': 'image', 'image': rgb})
    return content


def user_message(record, prompt=None, root=ROOT):
    content = image_content(record, root, label_slides=prompt is not None)
    if prompt is not None:
        content.insert(0, {'type': 'text', 'text': prompt})
    return {'role': 'user', 'content': content}


def target(record, guided):
    return record['annotated_transcript'] if guided else record['transcript']
