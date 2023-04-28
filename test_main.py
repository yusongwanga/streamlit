import unittest
import pandas as pd
from io import StringIO
from app import *

class TestApp(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv("linkedin-jobs-usa.csv")
        self.sample_text = "This is a sample text with some links https://www.google.com/ and emojis 😀🤔."
        self.expected_result = "This is a sample text with some links  and emojis ."
        self.expected_lower = ['this', 'is', 'a', 'sample', 'text', 'with', 'some', 'links', 'and', 'emojis']
        self.expected_pos_tags = [('This', 'DT'), ('sample', 'JJ'), ('text', 'NN'), ('links', 'NNS'), ('emojis', 'NNS')]

    def test_remove_URL(self):
        result = remove_URL(self.sample_text)
        self.assertEqual(result, self.expected_result)

    def test_remove_emoji(self):
        result = remove_emoji(self.sample_text)
        self.assertEqual(result, self.expected_result)

    def test_remove_html(self):
        result = remove_html(self.sample_text)
        self.assertEqual(result, self.expected_result)

    def test_remove_punct(self):
        result = remove_punct(self.sample_text)
        self.assertEqual(result, "This is a sample text with some links  httpswwwgooglecom and emojis 🤔")

    def test_remove_quotes(self):
        result = remove_quotes(self.sample_text)
        self.assertEqual(result, "This is a sample text with some links httpswwwgooglecom and emojis ")

    def test_remove_stopwords(self):
        result = remove_stopwords(self.expected_lower)
        self.assertEqual(result, ['sample', 'text', 'links', 'emojis'])

    def test_get_wordnet_pos(self):
        self.assertEqual(get_wordnet_pos('JJ'), wordnet.ADJ)
        self.assertEqual(get_wordnet_pos('VB'), wordnet.VERB)
        self.assertEqual(get_wordnet_pos('NN'), wordnet.NOUN)
        self.assertEqual(get_wordnet_pos('RB'), wordnet.ADV)
        self.assertEqual(get_wordnet_pos(''), wordnet.NOUN)

    def test_get_pos_tags(self):
        result = get_pos_tags(self.expected_lower)
        self.assertEqual(result, self.expected_pos_tags)

if __name__ == '__main__':
    unittest.main()