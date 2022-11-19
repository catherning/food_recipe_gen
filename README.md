# food_recipe_gen

Git of the project for the Food recipe generation with computational creativity

## Goals
- Create recipes (list of instructions) based on a list of ingredients, and ingredient pairings (from the KitcheNette dataset)
- Make an extensive analysis of the datasets, and predict the cuisine country origin (French, Japanese, ...) based on the ingredients
- Make a comparison of the different ML models used for the two goals above

## Installation

Apart from the packages in the requirements.txt file, also need NLTK
`sudo apt-get install python-nltk` on a unix environment

(Using conda env, but installing packages with pip, tested with python 3.6)

## Datasets used

### Recipe1M
http://pic2recipe.csail.mit.edu/
We only need the textual info from the dataset: the det_ingrs.json (Ingredient detections file on the website) and the layer1 json.

```layers1.json
{
  id: String,  // unique 10-digit hex string
  title: String,
  instructions: [ { text: String } ],
  ingredients: [ { text: String } ],
  partition: ('train'|'test'|'val'),
  url: String
}
```

```det_ingrs.json
{
  valid: [ bool ] // True if the ingredient in the same order is really an ingredient ? #TODO: check
  id: String,  // unique 10-digit hex string
  ingredients: [ { text: String } ],
}
```

## Steps 

### Data processing

``
You can use the run_data_processing.bat script that calls data_processing.py to process the Recipe1M dataset. It : 
1. Cleans and counts the words, including the ingredients in the dataset
2. Tokenizes words in the recipes
The ingredients are those detected in the det_ingrs.json file, crossed with the main layer1.json 


#### Parameters
Only --recipe1m_path is necessary
--threshold_ingrs : minimum ingr count threshold
--minnuminstrs : min number of instructions (sentences)
--maxnuminstr : max number of instructions (sentences)
--maxnumingrs : max number of ingredients
--minnumingrs : min number of ingredients
--minnumwords : minimum number of words in recipe (sum of the words in all instructions)
--forcegen : Force the generation of the vocabulary files, even if they exist in the target folder
TODO: --forcegen-all : seems redundant with forcegen

TODO: générer automatiquement la docu

#### Created files
- allingrs_count.pkl: Counter of the ingredients of the preprocessed dataset (filtered out the recipes with too few/many ingredients/instructions) 
- allwords_count : Counter of the words in the instructions of the preprocessed dataset (filtered out the recipes with too few/many ingredients/instructions). The instructions usually refer to the ingredients, so the words vocabulary should include the ingredients vocabulary, but that may not always be the case 

- recipe1m_vocab_ingrs.pkl : Vocabulary (class defined in '''utils.py''') of ingredients in the dataset, created from the ingredients counter
- recipe1m_vocab_words.pkl : Vocabulary of words in the recipe instructions in the dataset, created from the words counter
- recipe1m_[train/val/test].pkl: Datasets for the food recipe generation models
Structure : [
{'instructions': [cleaned instruction (remove instruction number, replace words using the "replace_dict_instrs" dict)], 
'tokenized': toks,
'ingredients': ingrs_list, 
'title': title}
]

### Predict ingredient pairing
Use the KitcheNette model to predict the ingredient pairing score of all ingredient combinations in the dataset generated at the previous step

### Recipe generation
The main.py file is the main file for training the models. It can resumes training from a saved model.
The different kinds of model are:
- Seq2seq: baseline Seq2Seq with an RNN encoder and decoder. The optimizer is Adam with the Cross entropy loss
- Seq2seqAtt: Seq2Seq with a classic attention mechanism
- Seq2seqIngrAtt: Seq2Seq with the Ingredient attention mechanism 
- Seq2seqIngrPairingAtt: Seq2Seq with the Ingredient attention mechanism which includes the score of ingredient pairing
- Seq2seqTitlePairing: Ingredient pairing attention mechanism and includes the title of the recipe as input
- Seq2seqCuisinePairing: Includes the cuisine of the recipe as input
- Seq2seqTrans: Uses a Transformer decoder model
- HierarchicalSeq2seq: Hierarchical Seq2seq baseline
- HierarchicalAtt
- HierarchicalIngrAtt
- HierarchicalIngrPairingAtt
- HierarchicalTitlePairing
- HierarchicalCuisinePairing