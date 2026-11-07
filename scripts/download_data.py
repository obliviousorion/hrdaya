"""Download the official open ECG-ID ZIP; preserve archive and SHA256 locally."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://physionet.org/static/published-projects/ecgiddb/ecg-id-database-1.0.0.zip'

EXPECTED_SHA256 = 'eae8df68ef808aa5eaf713815705e0479109ad6fcbce042182cb9045c7069616'


def verify_archive(archive):
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError(f'Archive SHA256 mismatch: {actual}; remove it and retry')


def verify_files(folder):
    count = 0
    for line in (folder / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        target = (folder / name.lstrip('*')).resolve()
        if not target.is_relative_to(folder.resolve()):
            raise ValueError('Unsafe checksum path')
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'File SHA256 mismatch: {name}')
        count += 1
    if count != 999:
        raise ValueError(f'Expected 999 checksummed files, found {count}')
    return count


def main():
    dest = ROOT / 'data'
    dest.mkdir(exist_ok=True)
    archive = dest / 'ecg-id-database-1.0.0.zip'
    if not archive.exists():
        part = archive.with_suffix('.part')
        try:
            with urllib.request.urlopen(URL, timeout=45) as r, part.open('wb') as f:
                while chunk := r.read(1024 * 1024):
                    f.write(chunk)
            with zipfile.ZipFile(part) as z:
                if z.testzip():
                    raise ValueError('ZIP checksum failure')
            verify_archive(part)
            part.replace(archive)
        finally:
            part.unlink(missing_ok=True)
    verify_archive(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError('Unsafe ZIP path')
        z.extractall(dest)
    verified = verify_files(dest / 'ecg-id-database-1.0.0')
    headers = list(dest.glob('**/Person_*/*.hea'))
    if len(headers) != 310:
        raise ValueError(f'Expected 310 headers; found {len(headers)}')
    manifest = {'url': URL, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                'records': len(headers), 'verified_files': verified, 'license': 'ODC-By-1.0',
                'source': 'https://physionet.org/content/ecgiddb/1.0.0/'}
    (dest / 'download.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        raise SystemExit(f'Dataset unavailable: {e}. No experimental result produced.')
