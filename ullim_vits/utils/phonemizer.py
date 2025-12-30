from g2pk2 import G2p
import re


class KoreanPhonemizer:
    def __init__(self):
        self.g2p = G2p()

        self.pad = '_'
        self.eos = '~'
        self.phonemes = [
            self.pad, self.eos,
            'ㄱ', 'ㄲ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
            'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
            'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
            ' ', '.', ',', '!', '?'
        ]

        self.symbol_to_id = {s: i for i, s in enumerate(self.phonemes)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.phonemes)}

    def __call__(self, text):
        return self.phonemize(text)

    def phonemize(self, text):
        text = self.clean_text(text)
        phonemes = self.g2p(text)
        return phonemes

    def clean_text(self, text):
        text = re.sub(r'[^\w\s,.!?가-힣]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def text_to_sequence(self, text):
        phonemes = self.phonemize(text)
        sequence = []
        for p in phonemes:
            if p in self.symbol_to_id:
                sequence.append(self.symbol_to_id[p])
            else:
                sequence.append(self.symbol_to_id[' '])
        return sequence

    def sequence_to_text(self, sequence):
        return ''.join([self.id_to_symbol.get(s, '') for s in sequence])

    @property
    def vocab_size(self):
        return len(self.phonemes)
