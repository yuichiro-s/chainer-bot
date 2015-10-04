import pytest

from bot.vocab import *


def test_reserved_ids():
    voc = Vocab()
    assert voc.eos_id == 0
    assert voc.unk_id == 1


def test_add_word():
    voc = Vocab()
    voc.add_word(u'a')
    voc.add_word(u'b')
    assert voc.get_id(u'a') == 2
    assert voc.get_word(2) == u'a'
    assert voc.get_id(u'b') == 3
    assert voc.get_word(3) == u'b'
    assert voc.get_id(u'c') == voc.unk_id
    assert voc.size() == 4


def test_raise_type_error_for_non_unicode():
    voc = Vocab()
    with pytest.raises(TypeError):
        voc.add_word('str')
    with pytest.raises(TypeError):
        voc.get_id('str')

