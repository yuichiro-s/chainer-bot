
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

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, '{}\t{}'.format(i, w.encode('utf-8'))

    @classmethod
    def load(cls, path):
        voc = Vocab()   # <EOS> and <UNK> are added at this point
        with open(path) as f:
            for line in f:
                es = line.split('\t')
                if len(es) == 2:
                    i, w = es
                    w = w.decode('utf-8')
                    if i == 0:
                        assert w == EOS
                    elif i == 1:
                        assert w == UNK
                    else:
                        # add word
                        voc.add_word(w)
        return voc
