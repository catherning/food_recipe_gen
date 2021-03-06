import os
import json
from multi_key_dict import multi_key_dict
from fractions import Fraction
from collections import namedtuple
from unidecode import unidecode
import re
from textblob import Word

Ingredient = namedtuple("Ingredient", "name amount plural")

# TODO: clean this file
# issue with some ingr that are in format -_word

# removed |cloves?|, that is also an ingredient
units = re.compile(
    r'^(cubes?|cups?|c(a?ns?)?|[tT]|bulbs?|sprigs?|glass(es)?|dice|blocks?|an?|l|fl(uid)?\.?|ears?|lea(f|ves)|jars?|cartons?|strips?|heads?|wedges?|envelopes?|pints?|stalks?|sticks?|pinch(es)?|qts?|quarts?|handful|weight|bottles?|grinds?|tb\.?|lbs?\.?|oz\.?|mls?|g|containers?|tablespoons?|teaspoons?|dash(es)?|pounds?|pinch|box(es)?|cans?|(milli)?lit[er]{2}s?|pkg\.?|pack(et)s?|packages?|whole|bars?|bags?|tbsps?\.?|tbs\.?|ts|tsps?\.?|ounces?|dash|pieces?|slices?|bunch(es)?|sticks?|fl\.?|gallons?|squares?|knobs?|grams?|kgs?|tub(es)?|kilograms?|tins?|%|drizzles?|splash(es)?|chunks?|inch(es)?)$')
number = re.compile(r'((\d+)?\.\d+|(\d+(/\d+)?-)?\d+(/\d+)?)')
# TODO: add "whole" to blacklist ?????
# added cloth, to blacklist
blacklist = {'','-',';','c.','.','of', 'and', '&amp;', 'or', 'some', 'many', 'few', 'couple', 'as', 'needed', 'plus', 'more', 'to', 'serve',
             'taste', 'x', 'in', 'cook', 'with', 'at', 'room', 'temperature', 'only', 'cover', 'length',
             'into', 'if', 'then', 'out', 'preferably', 'well', 'good', 'better', 'best', 'about', 'all-purpose', 'all',
             'purpose', 'recipe', 'ingredient', ')', '(', 'thick-', 'very', 'eating', 'lengthwise', 'each',
             'cloth','glue','glycerin','erythritol'}#,

states ={'sliced','stock','chopped','cooked','dry','fresh','grated','ground','halved','large','minced','optional','organic',
                'powder','skinned','trimmed','warmed','white'} # TODO: add attribute to Ingredient for state ? useful ????

parens = re.compile(r'[(\x97].*[)\x97]')
illegal_characters = re.compile(r'[‶″Â]')
cut_list = ['for', 'cut', 'such as']
replacements = {'yoghurt': 'yogurt', "'s": '', 'squeeze': 'squeezed',
                'aubergine': 'eggplant', 'self raising': 'self-raising', 'pitta': 'pita', 'chile': 'chili','chilli':'chili',
                'tomate':'tomato','zucker':'sugar'} # removed , 'olife': 'olive'

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
            raw_name = raw_name.split(splitter)
            if raw_name[0]=='':
                raw_name=raw_name[-1]
            else:
                raw_name=raw_name[0]
    raw_name = illegal_characters.sub("", raw_name)
    raw_name = unidecode(raw_name).replace(" ", "_").replace(',', '').strip()
    for replacement in replacements:
        raw_name = raw_name.replace(replacement, replacements[replacement])
    
    # Splits the full raw name in parts, and look at the type of each current one by one
    parts = raw_name.split("_")   
    parts.reverse()  # because pop(0) is terrible
    while parts:
        current = parts.pop()

        # current to leave out
        if current in blacklist or ":" in current:
            continue
        
        # The current is a number
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
        
        # The current is a unit
        elif units.match(current):
            if current in ('a', 'an'):
                if num == 0:
                    num = 1
            elif unit is None:
                unit = current

        # Current is a modifier
        # elif current in modifiers or current[-2:] in ('ly', 'ed'):
        #     adjectives.add(modifier_unifications.get(current, current))
        
        # Otherwise current is just a name of ingr
        else:
            lemma = lemmatize(current)
            if lemma != current:
                current = lemma
                plural = True
            if name:
                name += "_" + current
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
    
    if name in states:
        return None
    
    elif name:
        return Ingredient(
            name=name,
            amount=(num if num != 0 else None, unit, orig_num, orig_unit),
            # modifiers=list(adjectives),
            plural=plural,
        )
    else:
        return None

if __name__=="__main__":
    print(normalize_ingredient("2 cups cooked plain rice, or Mexican Green Rice, page ____"))