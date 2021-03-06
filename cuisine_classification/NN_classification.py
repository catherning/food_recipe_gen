import argparse
import csv
import itertools
import logging
import math
import os
import pathlib
import pickle
import sys
from collections import Counter
from datetime import datetime
from joblib import load, dump

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.getcwd()))

import recipe_1m_analysis.utils as utils
from recipe_1m_analysis.data_processing import cleanCounterIngr
from recipe_1m_analysis.ingr_normalization import normalize_ingredient
from recipe_gen.main import getDefaultArgs




# ## Data preprocessing
LOGGER = logging.getLogger()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


argparser = argparse.ArgumentParser()
argparser.register('type', 'bool', str2bool)

# Directories
argparser.add_argument('--data-folder', type=str, #default = os.path.join(os.getcwd(),"cuisine_classification","ml_results"),
                       help='Dataset path')
argparser.add_argument('--file-type', type=str, default="full", choices=["random", "cluster_centroid", "cluster_centroid10000","full"],
                       help='Type of undersampling. Full is no undersampling.')
argparser.add_argument('--saving-path', type=str, default=os.path.join(os.getcwd(), "cuisine_classification", "results"),
                       help='Saving path')
argparser.add_argument('--load-folder', type=str,
                       help='Loading best model in folder load-path')
argparser.add_argument('--classify-folder', type=str, default=os.path.join(os.getcwd(), "recipe_1m_analysis", "data"),
                       help='The folder where classify-file is.')
argparser.add_argument('--classify-file', type=str, default="train", choices=["train", "test", "val"],
                       help='The dataset of ingr to classify (train, test or dev)')

# Model settings
argparser.add_argument('--embedding-layer', type='bool', nargs='?',
                       const=True, default=False,
                       help='Enable training')
argparser.add_argument('--ingr-embed', type=int, default=300,
                       help='Display steps')
argparser.add_argument('--hidden-dim1', type=int, default=256,
                       help='Display steps')
argparser.add_argument('--hidden-dim2', type=int, default=64,
                       help='Display steps')
argparser.add_argument('--hidden-dim3', type=int, default=16,
                       help='Display steps')
argparser.add_argument('--balanced', type='bool', nargs='?',
                       const=True, default=False,
                       help='Weighted loss to give more importance to less represented cuisines. Default: False')

# Training settings
argparser.add_argument('--batch-size', type=int, default=64,
                       help='Display steps')
argparser.add_argument('--min-ingrs', type=int, default=10,
                       help='Display steps')
argparser.add_argument('--max-ingrs', type=int, default=50,
                       help='Display steps')
argparser.add_argument('--nb-epochs', type=int, default=300,
                       help='Display steps')
argparser.add_argument('--proba-threshold', type=float,
                       help='Threshold proba for test and inference')
argparser.add_argument('--clustering', type='bool', nargs='?',
                       const=True, default=False,
                       help='Does clustering for the vocab')

argparser.add_argument('--print-step', type=int, default=50,
                       help='Display steps')
argparser.add_argument('--train-mode', type='bool', nargs='?',
                       const=True, default=True,
                       help='Enable training')
argparser.add_argument('--test', type='bool', nargs='?',
                       const=True, default=True,
                       help='Enable test')
argparser.add_argument('--classify', type='bool', nargs='?',
                       const=True, default=False,
                       help='Enable classifying on Recipe1M')
argparser.add_argument('--load', type='bool', nargs='?',  # should be able to remove this line and just use load-folder ?
                       const=True, default=False,
                       help='Load a model already trained')
argparser.add_argument('--device', type=int, default=0,
                       help='CUDA device')


def init_logging(args):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

    # Create save folder
    args.saving_path = saving_path = os.path.join(args.saving_path, "{}{}_{}{}".format(
        datetime.now().strftime('%m-%d-%H-%M'), args.balanced*'_bal', args.file_type, args.clustering*'_clust'))

    if not os.path.isdir(saving_path):
        pathlib.Path(saving_path).mkdir(parents=True, exist_ok=True)
        print('...created ' + saving_path)

    logfile = logging.FileHandler(os.path.join(saving_path, 'log.txt'), 'w')

    logfile.setFormatter(fmt)
    LOGGER.addHandler(logfile)


def logParam(args):
    for k, v in args.defaults.items():
        try:
            if v is None or getattr(args, k) != v:
                LOGGER.info("{} = {}".format(
                    k, getattr(args, k)))
        except AttributeError:
            continue


def createVocab(args, df):
    counter_ingrs = Counter()
    # counter_ingrs.update([normalize_ingredient(ingr).name for ingr in itertools.chain.from_iterable(df.loc[:, "ingredients"]) if normalize_ingredient(ingr) is not None])
    # just try counter_ingrs.update(df.loc[:,"ingredients"])
    for ingredients in df.loc[:, "ingredients"]:
        ingr_list = []
        for ingr in ingredients:
            try:
                ingr_list.append(normalize_ingredient(ingr).name)
            except AttributeError:
                continue

        counter_ingrs.update(ingr_list)
        
    if args.clustering:
        counter_ingrs, cluster_ingrs = cleanCounterIngr(counter_ingrs)

        # Add the ingredients to the vocabulary.
        vocab_ingrs = utils.Vocabulary()
        vocab_ingrs.add_word("<pad>")
        vocab_ingrs.add_word("<unk>")
        idx = 2
        for word, cnt in counter_ingrs.items():
            if cnt >= args.min_ingrs:
                for ingr in cluster_ingrs[word]:
                    idx = vocab_ingrs.add_word(ingr, idx)
                idx += 1
                
    else:
        # TODO: add min_ingrs limit
        vocab_ingrs = utils.Vocabulary()
        vocab_ingrs.add_word("<pad>")
        vocab_ingrs.add_word("<unk>")
        
        for ingr, cnt in counter_ingrs.items():
            if cnt >= args.min_ingrs:
                vocab_ingrs.add_word(ingr)

    vocab_cuisine = utils.Vocabulary()
    for cuisine in df['cuisine'].value_counts().index:
        vocab_cuisine.add_word(cuisine)
            
    with open(os.path.join(args.saving_path,os.pardir, "vocab_ingr_"+ args.file_type + args.clustering*'_clust'+".pkl"), "wb") as f:
        pickle.dump(vocab_ingrs, f)
    
    with open(os.path.join(args.saving_path,os.pardir, "vocab_cuisine.pkl"), "wb") as f:
        pickle.dump(vocab_cuisine, f)

    return vocab_ingrs, vocab_cuisine


class IngrDataset(Dataset):
    """Recipes dataset for cuisine classification. Only from ingredients for now"""

    def __init__(self, args, input_, labels, vocab_ingrs, vocab_cuisine, type_label="cuisine"):
        """
        Args:
            file (string): Path to the file
        """
        self.args = args
        self.input_ = input_
        self.vocab_ingrs = vocab_ingrs
        self.vocab_cuisine = vocab_cuisine
        self.input_size = len(vocab_ingrs)
        self.max_ingr = args.max_ingrs
        self.labels = labels

        if type_label == "cuisine":
            self.processCuisine()
        elif type_label == "id":
            self.processId()
        else:
            raise AttributeError(
                "Don't know the type of label {}".format(type_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]

    def processId(self):
        # XXX: necessary ? id for classification for recipe1M..
        self.data = [None]*len(self.labels)
        for idx, (ingrs, label) in enumerate(zip(self.input_, self.labels)):
            self.data[idx] = (self.ingr2idx(ingrs), label)

    def processCuisine(self):
        self.data = {}
        for (idx, ingrs), label in zip(self.input_.iterrows(), self.labels.iterrows()):
            # try except if want to remove recipes with too many unk ingrs ?
            self.data[idx] = (self.ingr2idx(ingrs["ingredients"]),
                              self.vocab_cuisine.word2idx[label[1]["cuisine"]])

    def ingr2idx(self, ingr_list):
        if self.args.embedding_layer:
            input_ = [0]*self.max_ingr
            for i,ingr in enumerate(ingr_list):
                try:
                    input_[i]=self.vocab_ingrs.word2idx[ingr]
                except KeyError:
                    try:
                        input_[i]=self.vocab_ingrs.word2idx["<unk>"]
                    except IndexError:
                        break
                except IndexError:
                    break
            
            output = torch.LongTensor(input_)
            
        else:
            input_ = []
            for i,ingr in enumerate(ingr_list):
                try:
                    input_.append(self.vocab_ingrs.word2idx[ingr])
                except KeyError:
                    input_.append(self.vocab_ingrs.word2idx["<unk>"])
            
            output = torch.LongTensor(input_)
            onehot_enc = F.one_hot(output, self.input_size)
            output = torch.sum(onehot_enc, 0).float()

        # Removing samples where you don't know more than 2 of the ingr doesn't help the model much ?
        # if output[self.vocab_ingrs.word2idx["<unk>"]]>3:
        #     count_unk+=1
        #     raise AttributeError
        return output


def make_weights_for_balanced_classes(samples, vocab_cuisine):
    # from https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
    nclasses = len(vocab_cuisine.word2idx)
    count = [0] * nclasses
    weight_per_class = [0.] * nclasses
    N = len(samples)

    for a, el in samples.value_counts().items():
        count[vocab_cuisine.word2idx[a]] = el

    LOGGER.info("Class weights")
    # XXX to still sample the others, add log ? so goes back to N, not max(count)
    for i in range(nclasses):
        # divide by max count[i] ? Or just different scale, order is same
        weight_per_class[i] = math.log(N/float(count[i]))
        LOGGER.info("{} = {}".format(vocab_cuisine.idx2word[i], weight_per_class[i]))
    weight = [0] * N

    for idx, val in enumerate(samples):
        weight[idx] = weight_per_class[vocab_cuisine.word2idx[val]]

    return torch.Tensor(weight_per_class), torch.DoubleTensor(weight)

def createDataset(args, df):
    try:
        X_train,y_train,X_test,y_test,X_dev,y_dev = load(os.path.join(args.data_folder,"data_split.joblib"))
    except FileNotFoundError:    
        X_train, X_dev, y_train, y_dev = train_test_split(
            df["ingredients"], df["cuisine"], test_size=0.2, random_state=42, stratify=df["cuisine"])
        X_dev, X_test, y_dev, y_test = train_test_split(
            X_dev, y_dev, test_size=0.5, random_state=42, stratify=y_dev)

        X_train = X_train.reset_index()
        X_dev = X_dev.reset_index()
        X_test = X_test.reset_index()
        y_train = y_train.reset_index()
        y_dev = y_dev.reset_index()
        y_test = y_test.reset_index()
        
        dump((X_train,y_train,X_test,y_test,X_dev,y_dev),os.path.join(args.data_folder,"data_split.joblib"))
    
    return X_train,y_train,X_test,y_test,X_dev,y_dev


def createDataLoaders(args, vocab_ingrs, vocab_cuisine,X_train,y_train,X_test,y_test,X_dev,y_dev):

    train_dataset = IngrDataset(args,X_train, y_train, vocab_ingrs, vocab_cuisine)
    dev_dataset = IngrDataset(args,X_dev, y_dev, vocab_ingrs, vocab_cuisine)
    test_dataset = IngrDataset(args,X_test, y_test, vocab_ingrs, vocab_cuisine)

    if args.balanced:
        # Weighted random sampling, with stratified split for the train and test dataset. But loss doesn't decrease (need to see more epochs ?)
        weights_classes, weights_labels = make_weights_for_balanced_classes(
            y_train["cuisine"], vocab_cuisine)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights_labels, len(weights_labels))
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size)
        weights_classes = None

    dev_loader = DataLoader(dev_dataset, batch_size=1,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)
    return train_loader, dev_loader, test_loader, weights_classes

# # Model


class Net(nn.Module):
    def __init__(self, args, vocab_ingrs, vocab_cuisine, weights_classes=None):
        super(Net, self).__init__()
        self.args = args
        self.vocab_size = len(vocab_ingrs)
        self.num_classes = num_classes = len(vocab_cuisine)
        self.vocab_ingrs = vocab_ingrs
        self.vocab_cuisine = vocab_cuisine
        self.device = args.device
        ingr_embed = args.ingr_embed
        hidden_dim1 = args.hidden_dim1
        hidden_dim2 = args.hidden_dim2
        self.hidden_dim3 = hidden_dim3 = args.hidden_dim3

        if self.args.embedding_layer:
            self.embedding_layer=(self.vocab_size,ingr_embed)
            model = [nn.Linear(ingr_embed * args.max_ingr, hidden_dim1, bias=True)]
        else:
            model=[nn.Linear(self.vocab_size, hidden_dim1, bias=True)]        
        
        model+=[nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim1, hidden_dim2, bias=True),
                nn.ReLU(),
                ]

        if hidden_dim3:
            model+=[nn.Dropout(0.2),
                    nn.Linear(hidden_dim2, hidden_dim3, bias=True),
                         nn.ReLU(),
                        nn.Linear(hidden_dim3, num_classes, bias=True)]
        else:
            model.append(nn.Linear(
                hidden_dim2, num_classes, bias=True))

        self.model = nn.Sequential(*model)
        
        if args.balanced:
            self.criterion = nn.CrossEntropyLoss(weights_classes)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        if self.args.embedding_layer:
            x = self.embedding_layer(x)
            x = x.view(x.shape[0], -1)
  
        out = self.model(x)
        return out

    def train_process(self, train_loader, dev_loader, result_folder):
        LOGGER.info("Begin training")
        epoch_accuracy = []
        epoch_dev_accuracy = []
        best_score = 0

        for epoch in range(self.args.nb_epochs):
            self.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # print statistics
                running_loss += loss.item()
                if i % self.args.print_step == self.args.print_step-1:
                    LOGGER.info('[Epoch {}, Iteration {}] loss: {:.3f}'.format(
                        epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

                accuracy = 100 * correct / total
            epoch_accuracy.append(accuracy)

            LOGGER.info('Accuracy of the network on epoch {}: {:.3f}'.format(
                epoch+1, accuracy))

            dev_accuracy = self.test(dev_loader)
            epoch_dev_accuracy.append(dev_accuracy)
            if dev_accuracy > best_score:
                best_score = dev_accuracy
                LOGGER.info("Best model so far. Saving it.")
                torch.save(self.state_dict(), os.path.join(
                    result_folder, "best_model"))

        LOGGER.info('Finished Training')
        return loss, epoch_accuracy, epoch_dev_accuracy

    def test(self, dataloader, dataset_type="dev", threshold=None):
        self.eval()
        count_unk = 0
        correct = 0
        total = 0
        predictions = {}
        predictions_list = []            

        with torch.no_grad():
            for i,data in enumerate(dataloader):
                inputs = data[0].to(self.device)
                
                outputs = self.forward(inputs)

                if threshold:
                    proba = torch.nn.functional.softmax(outputs, dim=1)
                    p_max, predicted = torch.max(proba, 1)
                    if p_max < threshold:
                        continue
                else:
                    _, predicted = torch.max(outputs.data, 1)

                try:
                    labels = data[1].to(self.device)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                except AttributeError:
                    pass
                
                predictions_list.append(predicted[0].item())
                predictions[data[1][0]] = predicted[0].item()
        
        if threshold:
            LOGGER.info("Model confident for {}%% of recipes".format(len(predictions)/len(dataloader)*100))

        if dataset_type == "classify":
            return predictions, predictions_list
        else:
            try:
                accuracy = 100 * correct / total
                LOGGER.info('Accuracy of the network on the {} dataset: {:.3f}% for {} samples with threshold {}'.format(
                    dataset_type, accuracy, total, threshold))
                return accuracy
            except ZeroDivisionError:
                return 0
    
    def scikitEval(self,dataloader):
        enc = load(os.path.join("cuisine_classification","ml_results","scikit_vocab_cuisine.joblib"))
        _,y_pred = self.test(dataloader,"classify")
        y_test = [sample[1] for sample in dataloader.dataset.data.values()]
        res=[accuracy_score(y_test,y_pred),balanced_accuracy_score(y_test,y_pred),precision_score(y_test,y_pred,average='weighted'),recall_score(y_test,y_pred,average='weighted'),f1_score(y_test,y_pred,average='weighted')]
        print(res)

    def plotAccuracy(self, results_folder, epoch_accuracy, epoch_test_accuracy):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x_points = [i for i in range(0, self.args.nb_epochs)]

        ax.plot(x_points, epoch_accuracy, 'b')
        ax.plot(x_points, epoch_test_accuracy, 'b')
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.set_title('Accuracy during training')

        fig.savefig(os.path.join(results_folder,
                                 'accuracy_training.png'), dpi=fig.dpi)

    def saveResults(self, results_folder, loss, epoch, epoch_accuracy, epoch_test_accuracy, dev_accuracy, dev_accuracy_threshold):
        results_file = os.path.join(results_folder, "results.csv")
        with open(results_file, "w", newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["file", "balanced", "epoch", "train_accuracy",
                             "test_accuracy", "dev_accuracy", "threshold", "dev_accuracy_threshold"])
            writer.writerow([self.args.file_type, self.args.balanced, self.args.nb_epochs, epoch_accuracy[-1],
                             epoch_test_accuracy[-1], dev_accuracy, self.args.proba_threshold, dev_accuracy_threshold])

        torch.save(self.state_dict(), os.path.join(
            results_folder, "model_logweights"))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(results_folder, "training_state_logweights"))

    def classifyFromIngr(self):
        with open(os.path.join(self.args.classify_folder, "recipe1m_"+self.args.classify_file+".pkl"), "rb") as f:
            data = pickle.load(f)
        data_ingrs = [[ingr.name for ingr in v["ingredients"]] for v in data.values()]
        data_keys = list(data.keys())
        split_size = 50000
        for i in range(len(data_ingrs)//split_size+1):
            dataset = IngrDataset(self.args,data_ingrs[i*split_size:(i+1)*split_size], data_keys[i*split_size:(i+1)*split_size],
                                  self.vocab_ingrs, self.vocab_cuisine, type_label="id")
            print("Iter {} : Recipe1m loaded".format(i))
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            print("Classifying with proba {}...".format(self.args.proba_threshold))
            prob = self.args.proba_threshold
            predictions,_ = self.test(dataloader, dataset_type="classify",threshold=self.args.proba_threshold)

            for idx, prediction in predictions.items():
                data[idx]["cuisine"] = self.vocab_cuisine.idx2word[prediction]

        saving_file = os.path.join(self.args.classify_folder, "recipe1m_{}_cuisine_nn{}.pkl".format(self.args.classify_file,str(prob)*bool(prob)))
        with open(saving_file, "wb") as f:
            pickle.dump(data, f)
        LOGGER.info("Saving predictions to "+saving_file)


def main():
    args = getDefaultArgs(argparser)
    init_logging(args)
    logParam(args)

    df = pd.read_pickle(os.path.join(
        args.data_folder, args.file_type+"_data.pkl"))

    vocab_path = os.path.join(args.saving_path,os.pardir, "vocab_ingr_"+ args.file_type + args.clustering*'_clust'+".pkl")
    try:
        with open(vocab_path, "rb") as f:
            vocab_ingrs = pickle.load(f)
        
        with open(os.path.join(os.path.dirname(vocab_path),"vocab_cuisine.pkl"), "rb") as f:
            vocab_cuisine = pickle.load(f)
        LOGGER.info("Vocab loaded")
        
    except IOError:
        LOGGER.info("Create vocab")
        vocab_ingrs, vocab_cuisine = createVocab(args, df)

    HIDDEN_DIM3 = args.hidden_dim3
    
    X_train,y_train,X_test,y_test,X_dev,y_dev = createDataset(args, df)
        
    train_loader, dev_loader, test_loader, weights_classes = createDataLoaders(args,
        vocab_ingrs, vocab_cuisine,X_train,y_train,X_test,y_test,X_dev,y_dev)

    net = Net(args,vocab_ingrs, vocab_cuisine, weights_classes=weights_classes).to(args.device)
    
    if args.load:
        if args.train_mode:
            model = "model_logweights"
        elif args.classify or args.test:
            model = "best_model"

        args.load_folder = os.path.join(
            os.getcwd(), "cuisine_classification", "results", args.load_folder, model)
        # ,map_location='cuda:0'))
        net.load_state_dict(torch.load(args.load_folder))
        print("Network loaded.")

    if args.train_mode:
        loss, epoch_accuracy, epoch_test_accuracy = net.train_process(
            train_loader, dev_loader, args.saving_path)

        args.load_folder = os.path.join(args.saving_path, "best_model")
        net.load_state_dict(torch.load(args.load_folder))
        print("Best network reloaded.")

        test_accuracy = net.test(dev_loader, "test")
        test_accuracy_threshold = net.test(
            dev_loader, "test", 0.95)
        net.plotAccuracy(args.saving_path, epoch_accuracy, epoch_test_accuracy)
        net.saveResults(args.saving_path, loss, args.nb_epochs, epoch_accuracy,
                        epoch_test_accuracy, test_accuracy, test_accuracy_threshold)
        
        net.scikitEval(test_loader)

    if args.test and not args.train_mode:
        test_accuracy = net.test(test_loader, "test")
        test_accuracy_threshold = net.test(
            test_loader, "test", 0.95)
        print(test_accuracy,test_accuracy_threshold)
        
        net.scikitEval(test_loader)

    if args.classify:
        with open(os.path.join(args.classify_folder, "vocab_cuisine.pkl"), "wb") as f:
            pickle.dump(vocab_cuisine, f)
        net.classifyFromIngr()


if __name__ == "__main__":
    main()
