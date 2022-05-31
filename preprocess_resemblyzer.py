import argparse
from multiprocessing import cpu_count

import torch
import tqdm
from resemblyzer import preprocess_wav, VoiceEncoder

from utils.dsp import *
from utils.files import get_files, read_config
from utils.paths import Paths

parser = argparse.ArgumentParser(description='Preprocessing for WaveRNN and Tacotron')
parser.add_argument('--path', '-p', help='directly point to dataset path')
parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
args = parser.parse_args()


if __name__ == '__main__':
    config = read_config(args.config)
    wav_files = get_files(args.path, '.wav')
    wav_ids = {w.stem for w in wav_files}
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    voice_encoder = VoiceEncoder()
    dsp = DSP.from_config(config)

    for wav_file in tqdm.tqdm(wav_files, total=len(wav_files)):
        wav = preprocess_wav(wav_file)
        emb = voice_encoder.embed_utterance(wav)
        np.save(paths.speaker_emb / f'{wav_file.stem}.npy', emb, allow_pickle=False)

