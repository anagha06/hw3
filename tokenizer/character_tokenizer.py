from .tokenizer import Tokenizer

import torch

class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = "aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

        for i in range(len(self.characters)):
            self.vocab[self.characters[i]] = i
            
        raise NotImplementedError("Need to implement vocab initialization and the two functions from tokenizer")

    def encode(self, text):
        text = text.lower()
        indices = []
        for i in text:
            indices.append(self.characters.index(i))
        return torch.tensor(indices)

    
    def decode(self, word_arr):
        text = ""
        for char in word_arr:
            index = torch.argmax(char)
            text += self.characters[index]
        return text