
import re
import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
        self.unk_id = 0

    # Corpus를 pre-token 단위로 분리
    def pre_tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]+|\s+", text)

    # BPE Training
    def train(self, text, vocab_size=8000, min_frequency=2):

        pieces = Counter(self.pre_tokenize(text))

        # 각 pre-token을 character sequence로 변환
        sequences = {
            piece: list(piece)
            for piece in pieces
        }

        # 초기 character vocabulary
        vocab = set(
            char
            for piece in sequences.values()
            for char in piece
        )

        self.merges = []

        while len(vocab) < vocab_size:

            pair_counts = Counter()

            # 인접한 모든 pair의 빈도 계산
            for piece, symbols in sequences.items():
                freq = pieces[piece]

                for pair in zip(symbols, symbols[1:]):
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # 가장 빈도가 높은 pair 선택
            best_pair, frequency = max(
                pair_counts.items(),
                key=lambda item: (item[1], item[0])
            )

            if frequency < min_frequency:
                break

            left, right = best_pair
            merged = left + right

            self.merges.append(best_pair)
            vocab.add(merged)

            # 선택된 pair를 전체 sequence에서 병합
            for piece, symbols in sequences.items():

                new_symbols = []
                i = 0

                while i < len(symbols):

                    if (
                        i + 1 < len(symbols)
                        and symbols[i] == left
                        and symbols[i + 1] == right
                    ):
                        new_symbols.append(merged)
                        i += 2

                    else:
                        new_symbols.append(symbols[i])
                        i += 1

                sequences[piece] = new_symbols

        # Token → ID Dictionary 생성
        self.vocab = {"<unk>": self.unk_id}

        for idx, token in enumerate(sorted(vocab), start=1):
            self.vocab[token] = idx

        print("Vocabulary size:", len(self.vocab))
        print("Number of merges:", len(self.merges))

    # Text → Token IDs
    def encode(self, text):

        ids = []

        for piece in self.pre_tokenize(text):

            symbols = list(piece)

            # 학습된 merge rules를 순서대로 적용
            for left, right in self.merges:

                new_symbols = []
                i = 0

                while i < len(symbols):

                    if (
                        i + 1 < len(symbols)
                        and symbols[i] == left
                        and symbols[i + 1] == right
                    ):
                        new_symbols.append(left + right)
                        i += 2

                    else:
                        new_symbols.append(symbols[i])
                        i += 1

                symbols = new_symbols

            ids.extend(
                self.vocab.get(token, self.unk_id)
                for token in symbols
            )

        return ids

    # Token IDs → Text
    def decode(self, ids):

        reverse_vocab = {
            idx: token
            for token, idx in self.vocab.items()
        }

        return "".join(
            reverse_vocab.get(idx, "�")
            for idx in ids
        )

    # Save tokenizer
    def save(self, path):

        data = {
            "vocab": self.vocab,
            "merges": self.merges
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    # Load tokenizer
    def load(self, path):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = data["vocab"]
        self.vocab = {
            token: int(idx)
            for token, idx in self.vocab.items()
        }

        self.merges = [
            tuple(pair)
            for pair in data["merges"]
        ]
