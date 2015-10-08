from bot import corpus

import pytest


corpus_str = '''
1 2 1 3\t4 2 6
1 2 1 3\t4 2 6
1 2\t2 6
1 2 1\t4 2 6
1\t4 2 6
'''


@pytest.fixture(scope='session')
def corpus_path(tmpdir_factory):
    path = str(tmpdir_factory.mktemp('data').join('test_corpus'))
    with open(path, 'w') as f:
        print >> f, corpus_str
    return path


def test_load_corpus(corpus_path):
    cps = corpus.load_corpus(corpus_path)
    batches = corpus.create_batches(cps, 3)
    assert len(batches) == 2
