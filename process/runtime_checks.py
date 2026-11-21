"""Read ordinary runtime settings and verify their explicitly bound input files."""

import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    for filename, expected in settings['inputs_sha256'].items():
        if sha256(filename) != expected:
            raise ValueError(f'Runtime input changed: {filename}')
    return settings
