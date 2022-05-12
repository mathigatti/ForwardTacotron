import argparse
import itertools
import os
import subprocess
from pathlib import Path
from typing import Union

import torch
from torch import optim
from torch.nn import init
from torch.utils.data.dataloader import DataLoader

from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron
from trainer.common import to_device, np_now
from trainer.forward_trainer import ForwardTrainer
from utils.checkpoints import restore_checkpoint, init_tts_model
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.dsp import DSP
from utils.files import read_config
from utils.paths import Paths


def try_get_git_hash() -> Union[str, None]:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        print(f'Could not retrieve git hash! {e}')
        return None


def create_gta_features(model: Tacotron,
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path) -> None:
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, batch in enumerate(dataset, 1):
        batch = to_device(batch, device=device)

        with torch.no_grad():
            pred = model(batch)
        gta = pred['pitch']
        for j, item_id in enumerate(batch['item_id']):
            pred_pitch = gta[j][:, :batch['mel_len'][j]].squeeze()
            pred_pitch_norm = pred_pitch.softmax(0)
            pred_inds = torch.argmax(pred_pitch[1:, :], dim=0)
            pred_probs = torch.zeros(len(pred_inds))
            for k in range(len(pred_inds)):
                pred_probs[k] = pred_pitch_norm[pred_inds[k], k]
                if pred_probs[k] < 0.01:
                    pred_inds[k] = 0
            np.save(str(save_path/f'{item_id}.npy'), np_now(pred_inds.float()), allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ForwardTacotron TTS')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
    args = parser.parse_args()

    config = read_config(args.config)
    if 'git_hash' not in config or config['git_hash'] is None:
        config['git_hash'] = try_get_git_hash()
    dsp = DSP.from_config(config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    assert len(os.listdir(paths.alg)) > 0, f'Could not find alignment files in {paths.alg}, please predict ' \
                                           f'alignments first with python train_tacotron.py --force_align!'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    model = init_tts_model(config).to(device)
    print(f'\nInitialized tts model: {model}\n')
    optimizer = optim.Adam(model.parameters())
    restore_checkpoint(model=model, optim=optimizer,
                       path=paths.forward_checkpoints / 'latest_model.pt',
                       device=device)

    file =  '/Users/cschaefe/datasets/bild_snippets_cleaned/Snippets/r_0113_018.wav'
    wav = dsp.load_wav(file)

    mel = dsp.wav_to_mel(wav)
    mel = torch.tensor(mel).unsqueeze(0)
    spec = dsp.wav_to_spec(wav)

    pred = model({'mel': mel, 'x': torch.zeros(1), 'dur': torch.zeros(1), 'pitch': torch.zeros(1), 'mel_len': 1, 'energy': torch.zeros(1)})
    pred_pitch = pred['pitch'].squeeze().detach().cpu()

    pred_pitch_norm = pred_pitch.softmax(0)
    pred_inds = torch.argmax(pred_pitch[1:, :], dim=0)
    pred_probs = torch.zeros(len(pred_inds))
    for k in range(len(pred_inds)):
        pred_probs[k] = pred_pitch_norm[pred_inds[k], k]
        if pred_probs[k] < 0.001:
            pred_inds[k] = 0

    #pred_inds[pred_inds > 400] = 0

    pred_probs = torch.zeros(len(pred_inds))

    import matplotlib.pyplot as plt

    item_id = Path(file).stem
    pitch_gt = np.load(f'data/raw_pitch/{item_id}.npy')

    spec = np.flip(spec, axis=0)
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(np.log(spec[300:, -400:]), interpolation='nearest', aspect='auto')
    ax2.plot(pred_inds[-400:], color='blue')
    ax2.plot(pitch_gt[-400:], color='gray')
    plt.xlim(0, 400)
    plt.savefig('/tmp/pitch_gt.png')
    print(pred_inds)
    print(pitch_gt)