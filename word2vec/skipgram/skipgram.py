from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


setences = [
    "Kage is Teacher",
    "Mazong is Boss",
    "Niuzong is Boss",
    "Xiaobing is Student",
    "Xiaoxue is Student",
]


class SkipGram(nn.Module):
    def __init__(self, voc_size, embedding_size):
        super(SkipGram, self).__init__()
        self.input_to_hidden = nn.Linear(voc_size, embedding_size, bias=False)
        self.hidden_to_output = nn.Linear(embedding_size, voc_size, bias=False)

    def forward(self, X):
        hidden = self.input_to_hidden(X)
        output = self.hidden_to_output(hidden)
        return output


def train():
    # prepare data
    words = " ".join(setences).split()
    word_list = list(set(words))
    word_to_idx = {word: idx for idx, word in enumerate(word_list)}
    voc_size = len(word_list)
    skipgram_data = create_skipgram_dataset(setences=setences, window_size=2)

    # prepare model
    embedding_size = 2
    skipgram_model = SkipGram(voc_size=voc_size, embedding_size=embedding_size)

    lr = 1e-3
    epochs = 1000
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(skipgram_model.parameters(), lr=lr)

    loss_valus = []

    for epoch in range(epochs):
        loss_sum = 0
        for context, target in skipgram_data:
            X = one_hot_encoding(target, word_to_idx).float().unsqueeze(0)
            logger.debug(f"X={X}")
            y_true = torch.tensor([word_to_idx[context]], dtype=torch.long)
            y_pred = skipgram_model(X)
            logger.debug(f"pred={y_pred}")
            loss = criterion(y_pred, y_true)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 100 == 0:
            logger.info(f"Epoch:{epoch+1},Loss:{loss_sum/len(skipgram_data)}")
            loss_valus.append(loss_sum / len(skipgram_data))

    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(range(1, epochs // 100 + 1), loss_valus)
    plt.title("训练损失曲线")
    plt.xlabel("轮次")
    plt.ylabel("损失")
    plt.show()


def run():
    words = " ".join(setences).split()
    logger.debug(f"words:\n{words}")

    word_list = list(set(words))
    logger.debug(f"word list:\n{word_list}")

    word_to_idx = {word: idx for idx, word in enumerate(word_list)}
    logger.debug(f"word to index:\n{word_to_idx}")

    idx_to_word = {idx: word for idx, word in enumerate(word_list)}
    logger.debug(f"index to word:\n{idx_to_word}")

    voc_size = len(word_list)
    logger.debug(f"vocabulary size:{voc_size}")


def create_skipgram_dataset(setences, window_size=2):
    data = []
    for setence in setences:
        setence = setence.split()
        for idx, word in enumerate(setence):
            for neighbor in setence[
                max(idx - window_size, 0) : min(idx + window_size + 1, len(setence))
            ]:
                if neighbor != word:
                    data.append((neighbor, word))
    return data


def one_hot_encoding(word, word_to_idx):
    tensor = torch.zeros(len(word_to_idx))
    tensor[word_to_idx[word]] = 1
    return tensor


def test_one_hot_encoding():
    words = " ".join(setences).split()
    word_list = list(set(words))
    word_to_idx = {word: idx for idx, word in enumerate(word_list)}
    word_example = "Teacher"
    encoded = one_hot_encoding(word_example, word_to_idx)
    logger.debug(f"{word_example} encoded as {encoded}")


if __name__ == "__main__":
    # words = " ".join(setences).split()
    # word_list = list(set(words))
    # word_to_idx = {word: idx for idx, word in enumerate(word_list)}
    # data = create_skipgram_dataset(setences=setences)
    # logger.debug(data)
    # test_one_hot_encoding()
    # encoded_skip_grams = [
    #     (one_hot_encoding(context, word_to_idx), word_to_idx[target])
    #     for context, target in data
    # ]

    # for item in encoded_skip_grams:
    #     logger.debug(f"encoded skip grams: {item}")
    train()
