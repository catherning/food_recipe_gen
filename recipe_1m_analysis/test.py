import os
import json
from multi_key_dict import multi_key_dict
from fractions import Fraction
from collections import namedtuple
from unidecode import unidecode
import re
from textblob import Word

Ingredient = namedtuple("Ingredient", "name amount plural")

with open(os.path.join("D:\\Google Drive\\Catherning Folder\\THU\\Thesis\\Recipe datasets\\Recipe1M\\recipe1M_layers",
                       'det_ingrs.json'), 'r') as f:
    dets = json.load(f)

replace_dict_ingrs = {'and': ['&', "'n"], '': [
    '%', ',', '.', '#', '[', ']', '!', '?']}

units_dict = {
    "g": ["g", "lb", "ounce"],
    "teaspoon": ["teaspoon", "cup", "bottle"],
    "box": ["box", "can", "pack", "package"],
    "piece": ["piece", "slice"],
    "cm": ["cm", "inch"]
}

units = re.compile(
    r'^(cubes?|cups?|c(a?ns?)?|[tT]|bulbs?|sprigs?|glass(es)?|dice|blocks?|an?|l|fl(uid)?\.?|ears?|lea(f|ves)|jars?|cartons?|strips?|heads?|wedges?|envelopes?|pints?|stalks?|sticks?|pinch(es)?|qts?|quarts?|handful|weight|bottles?|grinds?|tb\.?|lbs?\.?|oz\.?|mls?|g|cloves?|containers?|tablespoons?|teaspoons?|dash(es)?|pounds?|pinch|box(es)?|cans?|(milli)?lit[er]{2}s?|pkg\.?|pack(et)s?|packages?|whole|bars?|bags?|tbsps?\.?|tbs\.?|ts|tsps?\.?|ounces?|dash|pieces?|slices?|bunch(es)?|sticks?|fl\.?|gallons?|squares?|knobs?|grams?|kgs?|tub(es)?|kilograms?|tins?|%|drizzles?|splash(es)?|chunks?|inch(es)?)$')
number = re.compile(r'((\d+)?\.\d+|(\d+(/\d+)?-)?\d+(/\d+)?)')
blacklist = {'of', 'and', '&amp;', 'or', 'some', 'many', 'few', 'couple', 'as', 'needed', 'plus', 'more', 'to', 'serve',
             'taste', 'x', 'in', 'cook', 'with', 'at', 'room', 'temperature', 'only', 'cover', 'length',
             'into', 'if', 'then', 'out', 'preferably', 'well', 'good', 'better', 'best', 'about', 'all-purpose', 'all',
             'purpose', 'recipe', 'ingredient', ')', '(', 'thick-', 'very', 'eating', 'lengthwise', 'each'}
parens = re.compile(r'[(\x97].*[)\x97]')
illegal_characters = re.compile(r'[‶″Â]')
cut_list = ['for', 'cut', 'such as']
replacements = {'yoghurt': 'yogurt', 'olife': 'olive', "'s": '', 'squeeze': 'squeezed',
                'aubergine': 'eggplant', 'self raising': 'self-raising', 'pitta': 'pita', 'chile': 'chili'}

unit_conversions = multi_key_dict()
unit_conversions['pounds', 'pound', 'lbs', 'lb'] = ('g', 453.6)
unit_conversions['ounces', 'ounce', 'ozs', 'oz', 'weight'] = ('g', 28.35)
unit_conversions['can', 'cans', 'cn'] = ('can', 1)
unit_conversions['pints', 'pint', 'pts', 'pt'] = ('l', 0.4732)
unit_conversions['quarts', 'quart', 'qts', 'qt'] = ('l', 1.946352946)
unit_conversions['cups', 'cup', 'c'] = ('l', 0.2366)
unit_conversions['cubes', 'cube'] = ('cube', 1)
unit_conversions['fluid', 'fl'] = ('l', 0.02957)
unit_conversions['tablespoons', 'tablespoon', 'tbsps',
                 'tbsp', 'tb', 'tbs', 'T'] = ('l', 0.01479)
unit_conversions['teaspoons', 'teaspoon',
                 'tsps', 'tsp', 't', 'ts'] = ('l', 0.004929)
unit_conversions['milliliters', 'millilitres', 'ml'] = ('l', 0.001)
unit_conversions['gram', 'gs', 'grams'] = ('g', 1)
unit_conversions['kilogram', 'kgs', 'kg', 'kilograms'] = ('g', 0.001)


# tb, ts, t, T, can, cans, cn, c,

# modifier_unifications = multi_key_dict()
# modifier_unifications['nonfat', 'non-fat'] = 'fat-free'
# modifier_unifications['low-fat', 'reduced-fat'] = 'fat-reduced'
# modifier_unifications['flatleaf'] = 'flat-leaf'


def get_ingredient(det_ingr, replace_dict):
    FLAG_DIGIT = False
    det_ingr_undrs = det_ingr['text'].lower()
    for i in det_ingr_undrs:
        if i.isdigit():
            FLAG_DIGIT = True
    det_ingr_undrs = ''.join(i for i in det_ingr_undrs if not i.isdigit())

    for rep, char_list in replace_dict.items():
        for c_ in char_list:
            if c_ in det_ingr_undrs:
                det_ingr_undrs = det_ingr_undrs.replace(c_, rep)
    det_ingr_undrs = det_ingr_undrs.strip()
    det_ingr_undrs = det_ingr_undrs.replace(' ', '_')

    return det_ingr_undrs, FLAG_DIGIT


def make_fraction(string):
    if "-" in string:
        return Fraction(string.split("-")[0])
    return Fraction(string)


def lemmatize(word):
    if word in {
        'flour',
        'pita',
        'pita',
        'chia',
        'asparagus',
        'couscous',
        'ricotta',
        'ciabatta',
        'pancetta',
        'pasta',
        'burrata',
        'bruschetta',
        'hummus',
        'acacia',
        'tilapia',
        'macadamia',
        'feta',
        'polenta',
        'stevia',
        'passata',
        'philadelphia',
        'salata',
        'your',
    }:
        return word
    if word.endswith("i") and word != "octopi":
        return word
    return Word(word).singularize()


def normalize_ingredient(raw_name):
    num, unit, adjectives = 0, None, set()
    name = None
    plural = False
    raw_name = parens.sub("", raw_name).replace("half", "1/2")
    raw_name = raw_name.lower()
    for splitter in cut_list:
        if splitter in raw_name:
            raw_name = raw_name.split(splitter)[0]
    raw_name = illegal_characters.sub("", raw_name)
    raw_name = unidecode(raw_name).replace("_", " ").replace(',', '').strip()
    for replacement in replacements:
        raw_name = raw_name.replace(replacement, replacements[replacement])
    parts = raw_name.split()
    parts.reverse()  # because pop(0) is terrible
    while parts:
        current = parts.pop()
        if current in blacklist or ":" in current:
            continue

        is_number = number.match(current)
        if is_number:
            if num == 0:
                if is_number.end() < len(current):
                    # unit included
                    num = make_fraction(current[:is_number.end()])
                    unit = current[is_number.end():]
                else:
                    num = make_fraction(current)
                    if parts and number.match(parts[-1]):  # next token is also a number
                        # then we add the numbers
                        next_token = parts.pop()
                        is_number = number.match(next_token)
                        if is_number.end() < len(next_token):
                            # next token has a number
                            num += make_fraction(next_token[:is_number.end()])
                            unit = next_token[is_number.end():]
                        else:
                            num += make_fraction(next_token)
        elif units.match(current):
            if current in ('a', 'an'):
                if num == 0:
                    num = 1
            elif unit is None:
                unit = current
        # elif current in modifiers or current[-2:] in ('ly', 'ed'):
        #     adjectives.add(modifier_unifications.get(current, current))
        else:
            lemma = lemmatize(current)
            if lemma != current:
                current = lemma
                plural = True
            if name:
                name += " " + current
            else:
                name = current
    num = float(num) if num else None
    if unit is not None and "/" in unit:
        unit = unit.split("/")[0]

    orig_num = num if num != 0 else None
    orig_unit = unit

    if unit in unit_conversions:
        unit, multiplier = unit_conversions[unit]
        if num is not None:
            num *= multiplier
    if name:
        return Ingredient(
            name=name,
            amount=(num if num != 0 else None, unit, orig_num, orig_unit),
            # modifiers=list(adjectives),
            plural=plural,
        )
    else:
        return None


for i in range(0, 10):
    det_ingrs = dets[i]['ingredients']

    for j, det_ingr in enumerate(det_ingrs):
        det_ingr_undrs, flag_digit = get_ingredient(
            det_ingr, replace_dict_ingrs)
        if flag_digit:
            # known_unit = False
            # for v in units_dict.values():
            #     for unit in v:
            #         if unit in det_ingr["text"]:
            #             known_unit = True
            #
            # if not known_unit:
            #     print(det_ingr)

            norm_ingr = normalize_ingredient(det_ingr["text"])
            print(norm_ingr.name)
            print(norm_ingr.amount)
            print(norm_ingr.plural)

