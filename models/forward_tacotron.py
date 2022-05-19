from pathlib import Path
from typing import Union, Callable, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Conv1d, LayerNorm, ReLU, Linear
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from models.common_layers import CBHG, LengthRegulator
from utils.text.symbols import phonemes


class SeriesPredictor(nn.Module):

    def __init__(self, num_chars, emb_dim=64, conv_dims=256, rnn_dims=64, dropout=0.5, out_dim=1):
        super().__init__()
        self.embedding = Embedding(num_chars, emb_dim)
        self.convs = torch.nn.ModuleList([
            BatchNormConv(emb_dim, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
            BatchNormConv(conv_dims, conv_dims, 5, relu=True),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, out_dim)
        self.dropout = dropout

    def forward(self,
                x: torch.Tensor,
                ada: torch.Tensor = None,
                alpha: float = 1.0) -> torch.Tensor:
        x = self.embedding(x)
        if ada is not None:
            x = x + ada
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.lin(x)
        return x / alpha


class PhonPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = BatchNormConv(80, 256, 3, relu=True)
        self.conv2 = BatchNormConv(256, 256, 3, relu=True)
        self.lin = Linear(256, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.lin(x)
        return x
        

class BatchNormConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel: int, relu: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=kernel // 2, bias=True)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        x = self.bnorm(x)
        return x


class ForwardTacotron(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 series_embed_dims: int,
                 num_chars: int,
                 durpred_conv_dims: int,
                 durpred_rnn_dims: int,
                 durpred_dropout: float,
                 pitch_conv_dims: int,
                 pitch_rnn_dims: int,
                 pitch_dropout: float,
                 pitch_strength: float,
                 energy_conv_dims: int,
                 energy_rnn_dims: int,
                 energy_dropout: float,
                 energy_strength: float,
                 rnn_dims: int,
                 prenet_dims: int,
                 prenet_k: int,
                 postnet_num_highways: int,
                 prenet_dropout: float,
                 postnet_dims: int,
                 postnet_k: int,
                 prenet_num_highways: int,
                 postnet_dropout: float,
                 n_mels: int,
                 padding_value=-11.5129):
        super().__init__()

        self.encoder = Sequential(
            nn.Conv1d(80, 256, 3, padding=1),
            nn.Conv1d(256, 32, 3, padding=1),
            nn.Conv1d(32, 1, 3, padding=1)
        )

        self.decoder = Sequential(
            nn.ConvTranspose1d(2, 32, 3, padding=1),
            nn.ConvTranspose1d(32, 256, 3, padding=1),
            nn.Conv1d(256, 80, 3, padding=1),
        )

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

    def __repr__(self):
        num_params = sum([np.prod(p.size()) for p in self.parameters()])
        return f'ForwardTacotron, num params: {num_params}'

    def forward(self, batch: Dict[str, torch.Tensor], train=True) -> Dict[str, torch.Tensor]:
        x = batch['x']
        mel = batch['mel']
        dur = batch['dur']
        mel_lens = batch['mel_len']

        if self.training:
            self.step += 1

        energy = mel.mean(dim=1).unsqueeze(1)
        pitch_pred = self.encoder(mel)
        conc = torch.cat([pitch_pred, energy], dim=1)
        mel_pred = self.decoder(conc)

        return {'pitch_pred': pitch_pred, 'mel_pred': mel_pred}

    def generate(self,
                 x: torch.Tensor,
                 alpha=1.0,
                 pitch_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
                 energy_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():

            ada_hat = self.phon_pred(x)
            ada_series = self.phon_series_lin(ada_hat)

            dur_hat = self.dur_pred(x, ada=ada_series, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x, ada=ada_series).transpose(1, 2)
            pitch_hat = pitch_function(pitch_hat)
            energy_hat = self.energy_pred(x, ada=ada_series).transpose(1, 2)
            energy_hat = energy_function(energy_hat)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat,
                                      ada_hat=ada_hat)

    @torch.jit.export
    def generate_jit(self,
                     x: torch.Tensor,
                     alpha: float = 1.0,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            dur_hat = self.dur_pred(x, alpha=alpha)
            dur_hat = dur_hat.squeeze(2)
            if torch.sum(dur_hat.long()) <= 0:
                torch.fill_(dur_hat, value=2.)
            pitch_hat = self.pitch_pred(x).transpose(1, 2) * beta
            energy_hat = self.energy_pred(x).transpose(1, 2)
            return self._generate_mel(x=x, dur_hat=dur_hat,
                                      pitch_hat=pitch_hat,
                                      energy_hat=energy_hat, ada_hat=None)

    def get_step(self) -> int:
        return self.step.data.item()

    def _generate_mel(self,
                      x: torch.Tensor,
                      dur_hat: torch.Tensor,
                      pitch_hat: torch.Tensor,
                      ada_hat: torch.Tensor,
                      energy_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.embedding(x)
        ada = self.phon_lin(ada_hat)
        x = x + ada
        x = x.transpose(1, 2)
        x = self.prenet(x)

        pitch_proj = self.pitch_proj(pitch_hat)
        pitch_proj = pitch_proj.transpose(1, 2)
        x = x + pitch_proj * self.pitch_strength

        energy_proj = self.energy_proj(energy_hat)
        energy_proj = energy_proj.transpose(1, 2)
        x = x + energy_proj * self.energy_strength

        x = self.lr(x, dur_hat)

        x, _ = self.lstm(x)

        x = self.lin(x)
        x = x.transpose(1, 2)

        x_post = self.postnet(x)
        x_post = self.post_proj(x_post)
        x_post = x_post.transpose(1, 2)

        return {'mel': x, 'mel_post': x_post, 'dur': dur_hat,
                'pitch': pitch_hat, 'energy': energy_hat}

    def _pad(self, x: torch.Tensor, max_len: int) -> torch.Tensor:
        x = x[:, :, :max_len]
        x = F.pad(x, [0, max_len - x.size(2), 0, 0], 'constant', self.padding_value)
        return x


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ForwardTacotron':
        model_config = config['forward_tacotron']['model']
        model_config['num_chars'] = len(phonemes)
        model_config['n_mels'] = config['dsp']['num_mels']
        return ForwardTacotron(**model_config)

    @classmethod
    def from_checkpoint(cls, path: Union[Path, str]) -> 'ForwardTacotron':
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        model = ForwardTacotron.from_config(checkpoint['config'])
        model.load_state_dict(checkpoint['model'])
        return model