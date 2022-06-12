import itertools
import pandas as pd
from singlish_dictionary import *



vowels = ['a', 'e', 'i', 'o', 'u']
non_vowels = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'w']


sin_df = pd.read_csv("Sinhala_dataset2.csv", header=None)
sin_df.columns = ['data']

sin_df["data"] = sin_df["data"].str.replace('?', '')
sin_df["data"] = sin_df["data"].str.replace('/', '')
sin_df["data"] = sin_df["data"].str.replace(',', '')
sin_df["data"] = sin_df["data"].str.replace('!', '')
sin_df["data"] = sin_df["data"].str.replace('.', '')
sin_df["data"] = sin_df["data"].str.replace('-', '')
sin_df["data"] = sin_df["data"].str.replace('#', '')
sin_df["data"] = sin_df["data"].str.replace('=', '')
sin_df["data"] = sin_df["data"].str.replace(')', '')
sin_df["data"] = sin_df["data"].str.replace('(', '')
sin_df["data"] = sin_df["data"].str.replace('\'', '')
sin_df["data"] = sin_df["data"].str.replace('\"', '')
sin_df["data"] = sin_df["data"].str.replace('\u200d', '')
sin_df["data"] = sin_df["data"].str.replace('\n', '')

tokens = set(sin_df['data'].str.lower().str.split().sum())

dic_keys = ['ක', 'ච', 'ට', 'ත', 'ප', 'ග', 'ජ', 'ඩ',
            'ද', 'බ', 'ය', 'ර', 'ල', 'ව', 'ස', 'ශ', 'හ', 'ල']
ka_dic = []
cha_dic = []
ta_dic = []
tha_dic = []
pa_dic = []
ga_dic = []
ja_dic = []
da_dic = []
dha_dic = []
ba_dic = []
ya_dic = []
ra_dic = []
la_dic = []
va_dic = []
sa_dic = []
sha_dic = []
ha_dic = []
la_dic = []
dic_list = [ka_dic, cha_dic, ta_dic, tha_dic, pa_dic, ga_dic, ja_dic, da_dic,
            dha_dic, ba_dic, ya_dic, ra_dic, la_dic, va_dic, sa_dic, sha_dic, ha_dic, la_dic]


dictionary = {}
num = 0
for item in dic_keys:
    dictionary[item] = dic_list[num]
    num += 1


count = -1
for key in dic_keys:
    count += 1
    print(count)
    dic_list[count].clear()

    for word in tokens:
        if word.startswith(key):
            dic_list[count].append(word)

    print(dic_list[count])
    print(len(dic_list[count]))

print('Dictionary List')
print(dic_list)


def edit_distance(string1, string2):

    if len(string1) > len(string2):
        difference = len(string1) - len(string2)
        string1[:difference]

    elif len(string2) > len(string1):
        difference = len(string2) - len(string1)
        string2[:difference]

    else:
        difference = 0

    for i in range(len(string1)):
        if string1[i] != string2[i]:
            difference += 1

    return difference


def transliterate(input_word):
    phonemes = []

    phoneme = ""
    count = 0

    for letter in input_word:
        if(letter in vowels):
            phoneme = phoneme+letter
            phonemes.pop()
            phonemes.append(phoneme)

        else:
            phoneme = letter
            phonemes.append(phoneme)
    print("Here is the phonemes")
    print(phonemes)

    posibilities = []
    for phoneme in phonemes:
        for key in singlish_dictionary:
            if(phoneme == key):
                posibilities.append(singlish_dictionary.get(phoneme))
                print(singlish_dictionary.get(phoneme))
                break
    print(posibilities)
   

    possible_words = []

    for combination in itertools.product(*posibilities):
        x = "".join(combination)
        print(x + "Tis is x value")
        possible_words.append(x)
        global starting_character
        starting_character = x[0]

   

    min = 100
    distance = 100
    closest_word = ''
    for item in possible_words:
        for word in dictionary[starting_character]:
            print(item + " "+word)
            if(len(item) < len(word)):
                print(edit_distance(item, word))
                distance = edit_distance(item, word)
            else:
                print(edit_distance(word, item))
                distance = edit_distance(word, item)
            if(min > distance):
                min = distance
                closest_word = item

    print(closest_word)
    return closest_word


