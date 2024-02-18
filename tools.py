import torch

from typing import List


def one_hot_encoding(word: str, word_to_idx: dict[str,int]) -> torch.Tensor:
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1
    return tensor


def prepare_words(sentences: List[str]) -> List[str]:
    return list(set((" ".join(sentences)).split()))


def prepare_word_to_idx(words: List[str]) -> dict[str,int]:
    return {word: idx for idx, word in enumerate(words)}


if __name__ == "__main__":
    corpus = ["Merlin is bird", "Tiger is cat"]

    words = prepare_words(corpus)
    print(words)

    word_to_idx = prepare_word_to_idx(words)
    print(word_to_idx)

    encoded = [(one_hot_encoding(w,word_to_idx),w) for w in words]
    print(encoded)
