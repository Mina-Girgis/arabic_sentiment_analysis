import re
import nltk
import pandas as pd
from regex import regex as re, regex
from textblob import *
from langdetect import detect
from nltk import wordpunct_tokenize, SnowballStemmer, ISRIStemmer, FreqDist
from translate import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
import string
# import emoji
import nltk
nltk.download('stopwords')
nltk.download('punkt')

class Preprocessing:
    punctuations_list = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
    arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)

    def __init__(self, data=''):
        self.text = data
        self.count = 0
        self.stop_words = self.read_stop_words()

    def read_stop_words(self):
        file_path = 'preprocessing/arabic_stop_words'
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read all lines and store them in a list
            file_content = [line.strip() for line in file.readlines()]
        return file_content

    def get_text(self):
        return self.text

    def remove_arabic_duplicate_letters(self):
        self.text = re.sub(r'([a-zA-Z\u0600-\u06FF])\1+', r'\1', self.text)
        return self

    def remove_english_duplicate_letters(self):
        cleaned_word = re.sub(r'(.)\1+', r'\1', self.text)
        self.text = cleaned_word
        return self

    def remove_english_words(self):
        sentences = self.text.split('.')
        # Identify and filter out English sentences
        non_english_sentences = []
        for sentence in sentences:
            try:
                if detect(sentence.strip()) != 'en':
                    non_english_sentences.append(sentence)
            except:
                pass

        # Reconstruct non-English text
        cleaned_text = '.'.join(non_english_sentences)
        remove_duplicate_text = self.remove_duplicate_letters(cleaned_text)
        self.text = remove_duplicate_text
        return self

    def remove_extra_spaces(self):
        # Use a regular expression to replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', self.text)
        # Strip leading and trailing spaces
        cleaned_text = cleaned_text.strip()
        self.text = cleaned_text
        return self

    def translate_sentence(self):
        translator = Translator(to_lang="ar")
        translated_sentence = translator.translate(self.text)
        self.text = translated_sentence
        return self

    def remove_arabic_stop_words(self):

        stop_words = self.stop_words
        words = wordpunct_tokenize(self.text)
        filtered_words = [word for word in words if word not in stop_words]

        # Join the filtered words back into a string
        filtered_text = ' '.join(filtered_words)
        self.text = filtered_text
        return self

    def remove_english_stop_words(self):
        stop_words = set(stopwords.words('english'))

        words = word_tokenize(self.text, language='english')

        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Join the filtered words back into a string
        filtered_text = ' '.join(filtered_words)
        self.text = filtered_text
        return self

    def fix_english_spelling_mistakes(self):
        data = TextBlob(self.text).correct()
        self.text = str(data)
        return self

    def remove_large_numbers(self):
        pattern = r'\d{3,}'
        text_without_large_numbers = re.sub(pattern, '', self.text)
        self.text = text_without_large_numbers
        return self

    def remove_all_numbers(self):
        pattern = r'\d'
        text_without_large_numbers = re.sub(pattern, '', self.text)
        self.text = text_without_large_numbers
        return self

    def remove_special_characters(self):
        # Define regex pattern to match non-emojis, Arabic and English characters
        non_emoji_pattern = re.compile(r"[^"
                                       u"\U0001F600-\U0001F64F"  # Emoticons
                                       u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                                       u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                                       u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                                       u"\U00002500-\U00002BEF"  # Chinese characters
                                       u"\U00002702-\U000027B0"
                                       u"\U00002702-\U000027B0"
                                       u"\U000024C2-\U0001F251"
                                       u"\U0001f926-\U0001f937"
                                       u"\U00010000-\U0010ffff"
                                       u"\u2640-\u2642"
                                       u"\u2600-\u2B55"
                                       u"\u200d"
                                       u"\u23cf"
                                       u"\u23e9"
                                       u"\u231a"
                                       u"\ufe0f"  # Dingbats
                                       u"\u3030"
                                       "a-zA-Z0-9\s"  # Alphanumeric characters and spaces
                                       "\u0600-\u06FF"  # Arabic characters
                                       "]+", flags=re.UNICODE)
        # Remove non-emojis from the text
        self.text = non_emoji_pattern.sub(r'', self.text)
        return self

    def remove_all_emojis(self):
        # Define a regular expression pattern to match emojis
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # Emoticons
                                   u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                                   u"\U0001F700-\U0001F77F"  # Alphabetic Presentation Forms
                                   u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                                   u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                                   u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                                   u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                                   u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                                   u"\U00002702-\U000027B0"  # Dingbats
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        # Use re.sub to replace all matches with an empty string
        self.text = emoji_pattern.sub('', self.text)
        return self

    def stem_text(self):
        stemmer = ISRIStemmer()
        words = word_tokenize(self.text)
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_text = ' '.join(stemmed_words)
        self.text = stemmed_text
        return self


    def remove_char(self):
        pattern = r'[^\p{Arabic}\p{Latin}\p{N}\p{So}\s]+'
        cleaned_text = re.sub(pattern, '', self.text)
        self.text = cleaned_text
        return self

    def get_word_frequencies(self):
        words = word_tokenize(self.text)
        fdist = FreqDist(words)
        print(fdist.most_common())
        return self

    # def convert_emoji_to_Text(self):
    #     text_copy = self.text
    #     # extract emojis
    #     pattern = r'[^\p{Emoji}]'
    #     cleaned_text = re.sub(pattern, '', text_copy)
    #     # convert it from string to list
    #     emoji_list = [char for char in cleaned_text]
    #     # translate emojis into english
    #     converted_text = emoji.demojize(cleaned_text)
    #     # replace every : into space
    #     converted_text = converted_text.replace(":", " ")  # Replace ":" with space
    #     # Use a regular expression to replace multiple spaces with a single space
    #     converted_text = re.sub(r'\s+', ' ', converted_text)
    #     # Strip leading and trailing spaces
    #     cleaned_converted_text = converted_text.strip()
    #     # convert it from string to list
    #     cleaned_converted_text = cleaned_converted_text.split()
    #
    #     # translate english into arabic
    #     translated_sentence_list = []
    #     for element in cleaned_converted_text:
    #         translator = Translator(to_lang="ar")
    #         translated_word = translator.translate(element)
    #         translated_sentence_list.append(translated_word)
    #     # put them together in map
    #     emoji_text_list = {key: value for key, value in zip(emoji_list, translated_sentence_list)}
    #     document_text = self.text
    #     for emojii, value in emoji_text_list.items():
    #         document_text = self.text.replace(emojii, value)
    #     self.text = document_text
    #     return self

    def normalize_arabic(self):
        self.text = re.sub("[إأآا]", "ا", self.text)
        self.text = re.sub("ى", "ي", self.text)
        self.text = re.sub("ؤ", "ء", self.text)
        self.text = re.sub("ئ", "ء", self.text)
        self.text = re.sub("ة", "ه", self.text)
        self.text = re.sub("گ", "ك", self.text)
        return self

    def remove_diacritics(self):
        self.text = re.sub(self.arabic_diacritics, '', self.text)
        return self

    def remove_punctuations(self):
        self.text = self.text.replace(self.punctuations_list, ' ')
        return self.text

    def preprocessing_methods(self, text):
        self.text = text

        self.remove_extra_spaces()
        # english
        # self.remove_english_duplicate_letters()
        # self.fix_english_spelling_mistakes()
        self.remove_english_stop_words()
        # self.translate_sentence()

        # arabic
        self.normalize_arabic()
        self.remove_diacritics()
        self.remove_punctuations()
        self.remove_char()
        self.remove_arabic_duplicate_letters()
        self.remove_arabic_stop_words()
        self.remove_special_characters()
        self.remove_all_numbers()
        self.remove_extra_spaces()
        self.remove_all_emojis()
        self.stem_text()
        # self.get_word_frequencies()


        self.count += 1
        # print(self.count)
        return self.text

# english
# arabic
# emoji
# numbers
# stop words
# special characters
