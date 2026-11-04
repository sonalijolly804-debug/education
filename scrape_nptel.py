import argparse
import asyncio
import csv
import re
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

FIELDS = ['course_id', 'lecture_number', 'lecture_name', 'language',
          'course_url', 'transcript_url', 'pdf_file']


def read_course_ids(course_ids, csv_path=None):
    values = list(course_ids)
    if csv_path:
        with Path(csv_path).open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if 'course_id' not in (reader.fieldnames or []):
                raise ValueError('The course CSV must contain a course_id column.')
            values.extend(row['course_id'].strip() for row in reader)
    values = list(dict.fromkeys(values))
    if not values or any(not re.fullmatch(r'\d+', value) for value in values):
        raise ValueError('Provide numeric NPTEL course IDs; course titles are not course IDs.')
    return values


async def collect_course(page, course_id, language, timeout):
    url = f'https://nptel.ac.in/courses/{course_id}'
    response = await page.goto(url, wait_until='domcontentloaded', timeout=timeout)
    if response is None or response.status >= 400:
        raise RuntimeError(f'Unable to open course {course_id}: HTTP {response.status if response else "unknown"}')
    await page.get_by_text(re.compile(r'^downloads$', re.I)).first.click(timeout=timeout)
    heading = page.get_by_text(re.compile(r'^transcripts$', re.I)).first
    await heading.wait_for(state='visible', timeout=timeout)
    section = heading.locator("xpath=ancestor::div[contains(@class, 'assignments')][1]")
    if not await section.count():
        section = heading.locator('xpath=ancestor::section[1]')
    if not await section.count():
        raise RuntimeError(f'Transcript section not found for {course_id}.')
    rows = section.locator('div.d-data')
    await rows.first.wait_for(state='visible', timeout=timeout)
    results = []
    for index in range(await rows.count()):
        row = rows.nth(index)
        name = (await row.inner_text()).split('Select Language', 1)[0].strip()
        dropdown = row.locator('.pseudo-input').first
        if not await dropdown.count():
            continue
        await dropdown.click(timeout=timeout)
        option = row.locator('li').filter(has_text=re.compile(rf'^\s*{re.escape(language)}\s*$', re.I)).first
        if not await option.count():
            await dropdown.click(timeout=timeout)
            continue
        await option.click(timeout=timeout)
        link = row.locator('a[href]').first
        await link.wait_for(state='visible', timeout=timeout)
        href = await link.get_attribute('href')
        resolved = urljoin(url, href or '')
        if not href or urlparse(resolved).scheme not in ('http', 'https'):
            raise ValueError(f'Invalid transcript link in course {course_id}, row {index + 1}.')
        number = re.search(r'\b(?:lecture|lec)\s*[-:.]?\s*(\d+)\b', name, re.I)
        results.append(dict(course_id=course_id,
                            lecture_number=number.group(1) if number else str(index + 1),
                            lecture_name=name, language=language, course_url=url,
                            transcript_url=resolved, pdf_file=''))
    if not results:
        raise RuntimeError(f'No {language} transcripts found for course {course_id}.')
    keys = [(r['lecture_number'], r['language']) for r in results]
    if len(keys) != len(set(keys)):
        raise ValueError(f'Duplicate lecture numbers on course {course_id}.')
    return results


def download_pdf(url, destination):
    import requests
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / 'transcript.pdf'
        if urlparse(url).hostname == 'drive.google.com':
            import gdown
            result = gdown.download(url=url, output=str(temporary), quiet=True, fuzzy=True)
            if not result:
                raise RuntimeError('Google Drive did not return a downloadable PDF.')
        else:
            with requests.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
        with temporary.open('rb') as handle:
            if b'%PDF-' not in handle.read(1024):
                raise ValueError('The response is not a PDF document.')
        temporary.replace(destination)


async def run(args, course_ids):
    from playwright.async_api import async_playwright
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                page.set_default_timeout(args.timeout * 1000)
                for course_id in course_ids:
                    records = await collect_course(page, course_id, args.language, args.timeout * 1000)
                    for record in records:
                        if args.download_dir:
                            safe_language = re.sub(r'[^A-Za-z0-9_-]', '_', args.language)
                            relative = Path(course_id) / f"lecture_{record['lecture_number']}_{safe_language}.pdf"
                            await asyncio.to_thread(download_pdf, record['transcript_url'], args.download_dir / relative)
                            record['pdf_file'] = relative.as_posix()
                        writer.writerow(record)
                        handle.flush()
                        if args.download_dir:
                            await asyncio.sleep(args.delay)
                    await asyncio.sleep(args.delay)
            finally:
                await browser.close()


def main():
    parser = argparse.ArgumentParser(description='Collect NPTEL lecture transcript links and optionally download PDFs.')
    parser.add_argument('--course-id', action='append', default=[])
    parser.add_argument('--courses', type=Path, help='CSV containing numeric course_id values.')
    parser.add_argument('--language', default='English-Verified')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download-dir', type=Path)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; choose a new output path.')
    if args.timeout <= 0 or args.delay < 0:
        parser.error('--timeout must be positive and --delay must be nonnegative.')
    try:
        course_ids = read_course_ids(args.course_id, args.courses)
    except ValueError as error:
        parser.error(str(error))
    asyncio.run(run(args, course_ids))


if __name__ == '__main__':
    main()
