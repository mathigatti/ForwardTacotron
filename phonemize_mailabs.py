from pathlib import Path

import tqdm
from dp.phonemizer import Phonemizer

from utils.files import get_files

phon = Phonemizer.from_checkpoint('/Users/cschaefe/stream_tts_models/phonemizer_en_rp_fix_er/model.pt')


def phonemize_csv(csv_file: Path):
    text_dict = {}
    with open(str(csv_file), encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines, total=len(lines)):
            try:
                split = line.split('|')
                item_id, text = split[0], split[-1]
                text = phon(text, lang='de')
                text_dict[item_id] = text
            except Exception as e:
                print(e)
    return text_dict


if __name__ == '__main__':

    csv_files = get_files('/Users/cschaefe/datasets/de_DE', extension='.csv')

    for csv_file in csv_files:
        name = csv_file.parent.stem
        print(name, csv_file)
        text_dict = phonemize_csv(csv_file)
        print([i for i in list(text_dict.items())[:10]])

        lines = [f'{a}|{b}\n' for a, b in text_dict.items()]
        with open(f'/tmp/metas/{name}.csv', 'w', encoding='utf-8') as f:
            f.writelines(lines)

    print()