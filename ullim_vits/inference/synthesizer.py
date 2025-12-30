import torch
from ullim_vits.models.vits.vits import VITS
from ullim_vits.utils.phonemizer import KoreanPhonemizer
from ullim_vits.utils.audio import save_wav
from ullim_vits.utils.logging import load_checkpoint


class Synthesizer:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VITS(config).to(self.device)

        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.phonemizer = KoreanPhonemizer()

    def synthesize(self, text, speaker_id=0, noise_scale=0.667, length_scale=1.0):
        with torch.no_grad():
            phonemes = self.phonemizer.text_to_sequence(text)
            phonemes = torch.LongTensor(phonemes).unsqueeze(0).to(self.device)
            phoneme_lengths = torch.LongTensor([phonemes.size(1)]).to(self.device)

            speaker_id_tensor = torch.LongTensor([speaker_id]).to(self.device) if speaker_id is not None else None

            audio, _, _, _ = self.model.infer(
                phonemes,
                phoneme_lengths,
                speaker_id=speaker_id_tensor,
                noise_scale=noise_scale,
                length_scale=length_scale
            )

            audio = audio.squeeze()

        return audio.cpu()

    def save_audio(self, audio, path):
        save_wav(path, audio, self.config.data.sampling_rate)
