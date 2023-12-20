import re
import emoji
from regex import regex as re
from nltk import wordpunct_tokenize, ISRIStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


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

    def remove_extra_spaces(self):
        # Use a regular expression to replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', self.text)
        # Strip leading and trailing spaces
        cleaned_text = cleaned_text.strip()
        self.text = cleaned_text
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

    def remove_all_numbers(self):
        pattern = r'\d'
        text_without_large_numbers = re.sub(pattern, '', self.text)
        self.text = text_without_large_numbers
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

    def convert_emoji_to_Text(self):
        # translate emojis into english
        converted_text = emoji.demojize(self.text)

        # replace every : into space
        converted_text = converted_text.replace(":", " ")  # Replace ":" with space
        converted_text = converted_text.replace("_", " ")  # Replace ":" with space

        # Use a regular expression to replace multiple spaces with a single space
        converted_text = re.sub(r'\s+', ' ', converted_text)
        # Strip leading and trailing spaces
        cleaned_converted_text = converted_text.strip()

        self.text = cleaned_converted_text
        return self

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
        self.remove_english_duplicate_letters()
        self.remove_english_stop_words()
        self.convert_emoji_to_Text()

        # arabic
        self.normalize_arabic()
        self.remove_diacritics()
        self.remove_punctuations()

        self.remove_char()
        self.remove_arabic_duplicate_letters()
        self.remove_arabic_stop_words()
        self.remove_all_numbers()
        self.remove_extra_spaces()
        self.stem_text()

        self.count += 1

        return self.text

# english
# arabic
# emoji
# numbers
# stop words
# special characters
