import sys
import os
sys.path.insert(0, os.path.join(os.getcwd()))
import argparse
import pandas as pd
import torch
import recipe_1m_analysis.utils as utils
import numpy as np
import math
from collections import Counter
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import pickle
from recipe_1m_analysis.data_processing import cleanCounterIngr

from sklearn import metrics
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ## Data preprocessing

FOLDER_PATH = "../recipe_datasets/" #"D:\\Google Drive\\Catherning Folder\\THU\\Thesis\\Recipe datasets\\" #"../recipe_datasets/" 
DATASET = ["scirep-cuisines-detail","Yummly28"]
FILES = ["random","cluster_centroid","full"]

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
argparser.add_argument('--data-folder', type=str, default=FOLDER_PATH,
                       help='Dataset path')
argparser.add_argument('--file-type', type=str, default="full", choices=FILES,
                       help='Type of undersampling. Full is no undersampling')
argparser.add_argument('--saving-path', type=str, default=os.path.join(os.getcwd(),"results"),
                       help='Saving path')
argparser.add_argument('--load-folder', type=str,
                       help='Loading best model in folder load-path')
argparser.add_argument('--classify-file', type=str, default=os.path.join(os.getcwd(),"recipe_1m_analysis","data","recipe1m_train.pkl"),
                       help='The dataset of ingr to classify')


# Model settings
argparser.add_argument('--print-step', type=int, default=512,
                       help='Display steps')
argparser.add_argument('--embed-dim1', type=int, default=128,
                       help='Display steps')
argparser.add_argument('--embed-dim2', type=int, default=32,
                       help='Display steps')
argparser.add_argument('--embed-dim3', type=int, default=50,
                       help='Display steps')
argparser.add_argument('--balanced', type='bool', nargs='?',
                        const=True, default=False,
                       help='Weighted loss to give more importance to less represented cuisines. Default: False')

# Training settings
argparser.add_argument('--batch-size', type=int, default=128,
                       help='Display steps')
argparser.add_argument('--min-ingrs', type=int, default=10,
                       help='Display steps')
argparser.add_argument('--nb-epochs', type=int, default=30,
                       help='Display steps')
argparser.add_argument('--proba-threshold', type=float, default=0.95,
                       help='Threshold proba for test and inference')
argparser.add_argument('--train-mode', type='bool', nargs='?',
                        const=True, default=True,
                       help='Enable training')
argparser.add_argument('--test', type='bool', nargs='?',
                        const=True, default=True,
                       help='Enable test')
argparser.add_argument('--classify', type='bool', nargs='?',
                        const=True, default=False,
                       help='Enable classifying on Recipe1M')
argparser.add_argument('--load', type='bool', nargs='?',
                        const=True, default=False,
                       help='Load a model already trained')
argparser.add_argument('--device', type=int, default=0,
                       help='CUDA device')

args = argparser.parse_args()

EMBED_DIM1 = args.embed_dim1
EMBED_DIM2 = args.embed_dim2
EMBED_DIM3 = args.embed_dim3
BATCH_SIZE = args.batch_size
PRINT_FREQ = args.print_step
NB_EPOCHS = args.nb_epochs
THRESHOLD = args.proba_threshold
balanced = args.balanced
load = args.load
if load:
    args.load_folder = os.path.join(os.getcwd(),"cuisine_classification","results",args.load_folder,"model_logweights")


def createDFrame(file):
    dataset = DATASET[1]
    df=pd.read_pickle(os.path.join(FOLDER_PATH,dataset,file+"_data.pkl"))
    df=df.set_index("id")

    dataset = DATASET[0]
    df2=pd.read_pickle(os.path.join(FOLDER_PATH,dataset,file+"_data.pkl"))
    df2["id"]=[len(df)+i for i in range(len(df2))]
    df2=df2.set_index("id")
    df = pd.concat([df, df2])

    return df

def createVocab(df,clustering=False):
    if clustering:
        counter_ingrs = Counter()
        counter_ingrs.update(df.loc[:,"ingredients"])
        # just try counter_ingrs.update(df.loc[:,"ingredients"])
        # for ingredients in df.loc[:,"ingredients"]:
        #     counter_ingrs.update([ingr for ingr in ingredients])
        counter_ingrs, cluster_ingrs = cleanCounterIngr(counter_ingrs)

        vocab_ingrs = utils.Vocabulary()
        idx = 0
        # Add the ingredients to the vocabulary.
        for word,cnt in counter_ingrs.items():
            if cnt >= args.min_ingrs:
                for ingr in cluster_ingrs[word]:
                    idx = vocab_ingrs.add_word(ingr, idx)
                idx += 1
        vocab_ingrs.add_word("<unk>",idx)

    else:
        vocab_ingrs = utils.Vocabulary()
        for ingredients in df.loc[:,"ingredients"]:
            for ingr in ingredients:
                vocab_ingrs.add_word(ingr)
        vocab_ingrs.add_word("<unk>")

    vocab_cuisine = utils.Vocabulary()
    for cuisine in df['cuisine'].value_counts().index:
        vocab_cuisine.add_word(cuisine)
        return vocab_ingrs, vocab_cuisine

class IngrDataset(Dataset):
    """Recipes dataset for cuisine classification. Only from ingredients for now"""

    def __init__(self, input_, labels,vocab_ingrs, vocab_cuisine,type_label="cuisine"):
        """
        Args:
            file (string): Path to the file
        """
        self.input_ = input_
        self.vocab_ingrs = vocab_ingrs
        self.vocab_cuisine = vocab_cuisine
        self.input_size = len(vocab_ingrs)
        self.labels = labels

        if type_label=="cuisine":
            self.processData()
        elif type_label=="id":
            self.processIngr()
        else:
            raise AttributeError("Don't know the type of label {}".format(type_label))

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
    
    def processIngr(self):
        self.data={}
        for (idx,ingrs),label in zip(self.input_.items(),self.labels.items()):
            self.data[idx]=(torch.LongTensor(self.ingr2idx(ingrs)),label[1])
            if idx==1000: #TODO: to remove when running smoothly
                break

    def processData(self):
        self.data={}
        for (idx,ingrs), label in zip(self.input_.iterrows(),self.labels.iterrows()):
            self.data[idx]=(self.ingr2idx(ingrs[1]["ingredients"]), self.vocab_cuisine.word2idx[label[1]["cuisine"]])
            if idx!=label[0]:
                raise AttributeError("Not same idx: {} {}".format(idx,label[0]))
    
    def ingr2idx(self, ingr_list):
        # If I didn't do the one-hot encoding by myself and used directly an embedding layer in the net, 
        # I would have to pad the input
        input_=[]
        for ingr in ingr_list:
            try:
                input_.append(self.vocab_ingrs.word2idx[ingr])
            except KeyError:
                input_.append(self.vocab_ingrs.word2idx["<unk>"])
        input_ = torch.LongTensor(input_)
        onehot_enc = F.one_hot(input_.to(torch.int64), self.input_size)
        output = torch.sum(onehot_enc,0)
        return output
    
        

def make_weights_for_balanced_classes(samples, vocab_cuisine):
    # from https://gist.github.com/srikarplus/15d7263ae2c82e82fe194fc94321f34e
    nclasses = len(vocab_cuisine.word2idx)
    count = [0] * nclasses
    weight_per_class = [0.] * nclasses
    N = len(samples)
    
    for a,el in samples.value_counts().items():
        count[vocab_cuisine.word2idx[a]]=el

    for i in range(nclasses): # XXX to still sample the others, add log ? so goes back to N, not max(count)
        weight_per_class[i] = math.log(N/float(count[i])) # divide by max count[i] ? Or just different scale, order is same
        print(vocab_cuisine.idx2word[i], weight_per_class[i])
    weight = [0] * N
    
    for idx, val in enumerate(samples): 
        weight[idx] = weight_per_class[vocab_cuisine.word2idx[val]] 
        
    return torch.Tensor(weight_per_class), torch.DoubleTensor(weight)


def createDataLoaders(df,vocab_ingrs,vocab_cuisine,balanced=False):
    #TODO when switch to python file, can put num_workers & have to put if __name__ == '__main__':
    X_train, X_dev, y_train, y_dev = train_test_split(df["ingredients"],df["cuisine"], test_size=0.2, random_state=42,stratify=df["cuisine"])
    X_dev, X_test, y_dev, y_test = train_test_split(X_dev,y_dev, test_size=0.5, random_state=42,stratify=y_dev)

    INPUT_SIZE = len(vocab_ingrs)

    X_train = X_train.reset_index()
    X_dev = X_dev.reset_index()
    X_test = X_test.reset_index()
    y_train = y_train.reset_index()
    y_dev = y_dev.reset_index()
    y_test = y_test.reset_index()

    train_dataset = IngrDataset(X_train,y_train,vocab_ingrs,vocab_cuisine)
    dev_dataset = IngrDataset(X_dev,y_dev,vocab_ingrs,vocab_cuisine)
    test_dataset = IngrDataset(X_test,y_test,vocab_ingrs,vocab_cuisine)

    if balanced:
        # Weighted random sampling, with stratified split for the train and test dataset. But loss doesn't decrease (need to see more epochs ?)
        weights_classes, weights_labels = make_weights_for_balanced_classes(y_train["cuisine"], vocab_cuisine) 
        print(len(weights_labels))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_labels, len(weights_labels)) 
        train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, sampler = sampler,num_workers=4)
    else:
        train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE,num_workers=4) 
        weights_classes = None

    dev_loader = DataLoader(dev_dataset,batch_size = BATCH_SIZE,num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE,num_workers=4)
    return train_loader, dev_loader, test_loader, weights_classes

# # Model
class Net(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim1, embedding_dim2, embedding_dim3=None):
        super(Net, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_dim3 = embedding_dim3

        self.layer_1 = nn.Linear(vocab_size, embedding_dim1, bias=True)
        self.layer_2 = nn.Linear(embedding_dim1, embedding_dim2, bias=True)
        if embedding_dim3:
            self.layer_3 = nn.Linear(embedding_dim2, embedding_dim3, bias=True)
            self.output_layer = nn.Linear(embedding_dim3, num_classes, bias=True)
        else:
            self.output_layer = nn.Linear(embedding_dim2, num_classes, bias=True)

    def forward(self, x):
        out = F.relu(self.layer_1(x))
        out = F.relu(self.layer_2(out))
        if self.embedding_dim3:
            out = F.relu(self.layer_3(out))
        out = self.output_layer(out)
        return out

def f2_score(y_true, y_pred, threshold=0.5):
    return fbeta_score(y_true, y_pred, 2, threshold)

def fbeta_score(y_true, y_pred, beta, threshold, eps=1e-9):
    beta2 = beta**2

    y_pred = torch.ge(y_pred.float(), threshold).float()
    y_true = y_true.float()

    true_positive = (y_pred * y_true).sum(dim=1)
    precision = true_positive.div(y_pred.sum(dim=1).add(eps))
    recall = true_positive.div(y_true.sum(dim=1).add(eps))

    return torch.mean(
        (precision*recall).
        div(precision.mul(beta2) + recall + eps).
        mul(1 + beta2))

def test_score(network, dataloader, vocab_ingrs, device=0,threshold=None):
    network.eval()
    count_unk=0
    correct = 0
    total = 0
    all_predict = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)

            # Removing samples where you don't know more than 2 of the ingr doesn't help the model much
            if inputs[0][vocab_ingrs.word2idx["<unk>"]]>3:
                count_unk+=1
                continue

            labels = data[1].to(device)
            outputs = network(inputs.float())
            
            if threshold:
                # Only taking the prediction when the model thinks it's threshold% probable that the label is x
                # Also not that good, accuracy of 42% instead of 82% on dev set 
                proba = torch.nn.functional.softmax(outputs,dim=1)
                p_max, predicted = torch.max(proba,1)
                if p_max < threshold:
                    continue
            else:
                _, predicted = torch.max(outputs.data, 1)
            
            all_predict.append(predicted)
            all_labels.append(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy= 100 * correct / total
    print('Accuracy of the network on the test dataset: {:.3f}% for {} samples'.format(accuracy,total))

    one_hot_pred = F.one_hot(torch.LongTensor(all_predict).to(torch.int64), network.num_classes)
    one_hot_lab = F.one_hot(torch.LongTensor(all_labels).to(torch.int64), network.num_classes)
    fbeta_pytorch = f2_score(one_hot_pred, one_hot_lab)

    # print('Score is {:.3f}%'.format(100* fbeta_pytorch))
    # print('Count unknown ingr: {}'.format(count_unk))
    
    return accuracy, fbeta_pytorch

def train(net,train_loader,dev_loader, vocab_ingrs, result_folder, weights_classes=None,device=0):
    if balanced:
        criterion = nn.CrossEntropyLoss(weights_classes)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    print("Begin training")
    # else:
    epoch_accuracy = []
    epoch_test_accuracy = []
    best_score = 0

    for epoch in range(NB_EPOCHS):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % PRINT_FREQ == PRINT_FREQ-1:    # print every 2000 mini-batches
                print('[Epoch {}, Iteration {}] loss: {:.3f}'.format(epoch + 1,i + 1,running_loss / 2000))
                running_loss = 0.0

            accuracy = 100 * correct / total
        epoch_accuracy.append(accuracy)
        
        print('Accuracy of the network on epoch {}: {:.3f}'.format(epoch+1,accuracy))
        
        dev_accuracy, dev_fscore = test_score(net,dev_loader,vocab_ingrs, device)
        epoch_test_accuracy.append(dev_fscore)
        if dev_fscore > best_score:
            best_score = dev_fscore
            print("Best model so far. Saving it.")
            torch.save(net.state_dict(), os.path.join(result_folder,"best_model"))

    print('Finished Training')
    return loss,epoch,epoch_accuracy,epoch_test_accuracy, optimizer


def plotAccuracy(results_folder, epoch_accuracy,epoch_test_accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_points = [i for i in range(0,NB_EPOCHS)]

    ax.plot(x_points, epoch_accuracy, 'b')
    ax.plot(x_points, epoch_test_accuracy, 'b')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy during training')
    # fig.show()

    fig.savefig(os.path.join(results_folder,'accuracy_training.png'), dpi=fig.dpi)

def saveResults(results_folder, net, loss, epoch, epoch_accuracy, epoch_test_accuracy, dev_fscore, dev_fscore_threshold, optimizer):
    results_file = os.path.join(results_folder,"results.csv")
    if os.path.isfile(results_file):
        with open(results_file,"w", newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["file","balanced","epoch","train_accuracy","test_fscore","dev_fscore","threshold","dev_fscore_threshold"])
            writer.writerow([args.file_type,balanced,NB_EPOCHS,epoch_accuracy[-1],epoch_test_accuracy[-1],dev_fscore,THRESHOLD,dev_fscore_threshold])
    else:
        with open(results_file,"a", newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([args.file_type,balanced,NB_EPOCHS,epoch_accuracy[-1],epoch_test_accuracy[-1],dev_fscore,THRESHOLD,dev_fscore_threshold])

    torch.save(net.state_dict(), os.path.join(results_folder,"model_logweights"))

    torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, os.path.join(results_folder,"training_state_logweights"))


def classifying(network, dataloader, vocab_ingrs, device=0, threshold=None):
    network.eval()
    count_unk=0
    predictions = {}
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            inputs = data[0].to(device)

            # Removing samples where you don't know more than 2 of the ingr doesn't help the model much
            # Can only do that if batch=1 OR do it in creation of IngrDataset
            if inputs[0][vocab_ingrs.word2idx["<unk>"]]>3:
                count_unk+=1
                continue

            outputs = network(inputs.float())
            
            if threshold:
                # Only taking the prediction when the model thinks it's threshold% probable that the label is x
                # Also not that good, accuracy of 42% instead of 82% on dev set 
                proba = torch.nn.functional.softmax(outputs,dim=1)
                p_max, predicted = torch.max(proba,1)
                if p_max < threshold:
                    continue
            else:
                _, predicted = torch.max(outputs.data, 1)

            for pred in predicted:
                predictions[data[1]]=pred
    return predictions

def classifyFromIngr(net,vocab_ingrs,vocab_cuisine):
    with open(args.classify_file,"rb") as f:
        data = pickle.load(f)
    df = pd.DataFrame(data)
    dataset = IngrDataset(df["ingredients"],df["id"],vocab_ingrs,vocab_cuisine,type_label="id")
    dataloader = DataLoader(dataset,batch_size = 1,shuffle=False)#,num_workers=4)
    predictions = classifying(net, dataloader, vocab_ingrs, device=args.device, threshold=THRESHOLD)


def main():
    df = createDFrame(args.file_type)
    vocab_ingrs, vocab_cuisine = createVocab(df)# TODO: when classification is done without. to test ,clustering = True)
    
    INPUT_SIZE = len(vocab_ingrs)
    NUM_CLASSES = len(vocab_cuisine)

    date = datetime.now().strftime("%m-%d-%H-%M")
    RESULTS_FOLDER = os.path.join(os.getcwd(),"results","nn_{}{}_{}".format(date,balanced*'_bal',args.file_type))
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    net = Net(INPUT_SIZE, NUM_CLASSES, EMBED_DIM1, EMBED_DIM2).to(args.device)

    if load:
        net.load_state_dict(torch.load(args.load_folder))
        
    if args.train_mode:
        train_loader, dev_loader, test_loader, weights_classes = createDataLoaders(df, vocab_ingrs,vocab_cuisine, balanced)
        loss, epoch, epoch_accuracy, epoch_test_accuracy, optimizer = train(net, train_loader, test_loader, vocab_ingrs, RESULTS_FOLDER, weights_classes, args.device)
        _, dev_fscore = test_score(net, dev_loader, vocab_ingrs, args.device)
        _, dev_fscore_threshold = test_score(net, dev_loader, vocab_ingrs, args.device, threshold=THRESHOLD)
        plotAccuracy(RESULTS_FOLDER, epoch_accuracy,epoch_test_accuracy)
        saveResults(RESULTS_FOLDER, net, loss, epoch, epoch_accuracy, epoch_test_accuracy, dev_fscore, dev_fscore_threshold,optimizer)

    if args.test:
        train_loader, dev_loader, test_loader, weights_classes = createDataLoaders(df, vocab_ingrs,vocab_cuisine, balanced)
        _, dev_fscore = test_score(net, dev_loader, vocab_ingrs, args.device)
        _, dev_fscore_threshold = test_score(net, dev_loader, vocab_ingrs, args.device, threshold=THRESHOLD)
    
    if args.classify:
        classifyFromIngr(net,vocab_ingrs,vocab_cuisine)


if __name__=="__main__":
    main()