import torch

sentence = "The New York Times is a daily newspaper based in New York City"
window_size = 2

voc = list(set(sentence.split()))

print(f"vocabular is {voc}")

voc_size = len(voc)
print(f"voc size is {voc_size}")

word_to_idx = {word:idx for idx,word in enumerate(voc)}
print(f"word to idx is: {word_to_idx}")

splitted = sentence.split()
data = []
for idx,word in enumerate(splitted):
    print(f"idx:{idx},word:{word}")
    for neighbor in splitted[max(idx-window_size,0):min(idx+window_size+1,len(splitted))]:
        if neighbor != word:
            print((neighbor,word))



