
EOS = u'<EOS>'
UNK = u'<UNK>'


class Vocab(object):
    """Mapping between words and IDs."""

    def __init__(self):
        self.w2i = {EOS: 0, UNK: 1}
        self.i2w = [EOS, UNK]
        self.eos_id = 0
        self.unk_id = 1

    def add_word(self, word):
        if not isinstance(word, unicode):
            raise TypeError
        if word not in self.w2i:
            new_id = self.size()
            self.w2i[word] = new_id
            self.i2w.append(word)

    def get_id(self, word):
        """Convert word to ID.
        Non-registered word has an ID for <UNK>"""
        if not isinstance(word, unicode):
            raise TypeError
        return self.w2i.get(word, self.unk_id)

    def get_word(self, id):
        """Convert ID to word."""
        return self.i2w[id]

    def size(self):
        return len(self.i2w)
