from utils.files import get_files
from pathlib import Path
from typing import Union


def ljspeech(path: Union[str, Path]):
    csv_files = get_files(path, extension='.csv')
    text_dict = {}
    speaker_dict = {}
    for csv_file in csv_files:
        name = csv_file.stem
        speaker_dict[name] = []
        with open(str(csv_file), encoding='utf-8') as f:
            for line in f:
                split = line.split('|')
                text_dict[split[0]] = split[-1]
                speaker_dict[split[0]] = name
    return text_dict, speaker_dict