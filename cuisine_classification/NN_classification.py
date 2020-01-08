import sys
import os
sys.path.insert(0, os.path.join(os.getcwd()))
import pandas as pd
import torch
import recipe_1m_analysis.utils as utils
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime
import csv

from sklearn import metrics
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader, Sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ## Data preprocessing

FOLDER_PATH = "D:\\Google Drive\\Catherning Folder\\THU\\Thesis\\Recipe datasets\\"
DATASET = ["scirep-cuisines-detail","Yummly28"]
FILES = ["random","cluster_centroid","full"]

EMBED_DIM1 = 300
EMBED_DIM2 = 64
BATCH_SIZE = 100
PRINT_FREQ = 20
NB_EPOCHS = 30

threshold = 0.90

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

def createVocab(df):
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

    def __init__(self, input_,labels, vocab_ingrs, vocab_cuisine,input_size):
        """
        Args:
            file (string): Path to the file
        """
        self.input_ = input_
        self.labels = labels
        self.vocab_ingrs = vocab_ingrs
        self.vocab_cuisine = vocab_cuisine
        self.input_size = input_size

        self.processData()

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
    
    def processData(self):
        self.data={}
        for ingrs, label in zip(self.input_.iterrows(),self.labels.iterrows()):
            self.data[ingrs[0]]=(self.ingr2idx(ingrs[1]["ingredients"]), self.vocab_cuisine.word2idx[label[1]["cuisine"]])
            if ingrs[0]!=label[0]:
                raise AttributeError("Not same idx: {} {}".format(ingrs[0],label[0]))
    
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

    train_dataset = IngrDataset(X_train,y_train,vocab_ingrs,vocab_cuisine,INPUT_SIZE)
    dev_dataset = IngrDataset(X_dev,y_dev,vocab_ingrs,vocab_cuisine,INPUT_SIZE)
    test_dataset = IngrDataset(X_test,y_test,vocab_ingrs,vocab_cuisine,INPUT_SIZE)

    if balanced:
        # Weighted random sampling, with stratified split for the train and test dataset. But loss doesn't decrease (need to see more epochs ?)
        weights_classes, weights_labels = make_weights_for_balanced_classes(y_train["cuisine"], vocab_cuisine) 
        print(len(weights_labels))
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_labels, len(weights_labels)) 
        train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE, sampler = sampler)#, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset,batch_size = BATCH_SIZE)#, pin_memory=True)    
        weights_classes = None

    dev_loader = DataLoader(dev_dataset,batch_size = 1)
    test_loader = DataLoader(test_dataset, batch_size=1)#, sampler = sampler)
    return train_loader, dev_loader, test_loader, weights_classes

# # Model
class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim1, embedding_dim2, num_classes):
        super(Net, self).__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.layer_1 = nn.Linear(vocab_size, embedding_dim1, bias=True)
        self.layer_2 = nn.Linear(embedding_dim1, embedding_dim1, bias=True)
        self.output_layer = nn.Linear(embedding_dim1, num_classes, bias=True)

    def forward(self, x):
        out = F.relu(self.layer_1(x))
        out = F.relu(self.layer_2(out))
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

def test_score(network, dataloader, vocab_ingrs, test=False,threshold=0.9):
    network.eval()
    count_unk=0
    correct = 0
    total = 0
    all_predict = []
    all_labels = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0]

            # Removing samples where you don't know more than 2 of the ingr doesn't help the model much
            if inputs[0][vocab_ingrs.word2idx["<unk>"]]>3:
                count_unk+=1
                continue

            labels = data[1]
            outputs = network(inputs.float())
            
            if test:
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
    print('Accuracy of the network on the test dataset: {.3f}% for {} samples'.format(accuracy,total))

    one_hot_pred = F.one_hot(torch.LongTensor(all_predict).to(torch.int64), network.num_classes)
    one_hot_lab = F.one_hot(torch.LongTensor(all_labels).to(torch.int64), network.num_classes)
    fbeta_pytorch = f2_score(one_hot_pred, one_hot_lab)

    print('Score is {.3f}%'.format(100* fbeta_pytorch))
    print('Count unknown ingr: {}'.format(count_unk))
    
    return accuracy, fbeta_pytorch

def train(net,train_loader,dev_loader, vocab_ingrs, result_folder, load=False, weights_classes=None):
    if balanced:
        criterion = nn.CrossEntropyLoss(weights_classes)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01) #change to Adam ?

    if load:
        net.load_state_dict(torch.load(os.path.join(FOLDER_PATH,DATASET[1],"model_logweights")))

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
            inputs = data[0]
            labels = data[1]
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
                print('[Epoch {}, Iteration {}] loss: {.3f}'.format(epoch + 1,i + 1,running_loss / 2000))
                running_loss = 0.0

            accuracy = 100 * correct / total
        epoch_accuracy.append(accuracy)
        
        print('Accuracy of the network on epoch {}: {.3f}'.format(epoch+1,accuracy))
        
        dev_accuracy, dev_fscore = test_score(net,dev_loader,vocab_ingrs, net.num_classes)
        epoch_test_accuracy.append(dev_fscore)
        if dev_fscore > best_score:
            best_score = dev_fscore
            torch.save(net.state_dict(), os.path.join(result_folder,"best_model"))

    print('Finished Training')
    return loss,epoch,epoch_accuracy,epoch_test_accuracy, optimizer


def plotAccuracy(results_folder, epoch_accuracy,epoch_test_accuracy):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_points = [i for i in range(0,NB_EPOCHS)]

    p = ax.plot(x_points, epoch_accuracy, 'b')
    p2 = ax.plot(x_points, epoch_test_accuracy, 'b')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_title('Accuracy during training')
    fig.show()

    fig.savefig(os.path.join(results_folder,'accuracy_training.png'), dpi=fig.dpi)

def saveResults(results_folder, net, loss, epoch, epoch_accuracy, epoch_test_accuracy, dev_fscore, optimizer):
    results_file = os.path.join(results_folder,"results.csv")
    if os.path.isfile(results_file):
        with open(results_file,"w", newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(["file","balanced","epoch","train_accuracy","test_fscore","dev_fscore","threshold"])
            writer.writerow([file,balanced,NB_EPOCHS,epoch_accuracy[-1],epoch_test_accuracy[-1],dev_fscore,threshold])
    else:
        with open(results_file,"a", newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([file,balanced,NB_EPOCHS,epoch_accuracy[-1],epoch_test_accuracy[-1],dev_fscore,threshold])

    torch.save(net.state_dict(), os.path.join(results_folder,"model_logweights"))

    torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, os.path.join(results_folder,"training_state_logweights"))

def main(argv, file, balanced, load):

    df = createDFrame(file)
    vocab_ingrs, vocab_cuisine = createVocab(df)
    
    INPUT_SIZE = len(vocab_ingrs)
    NUM_CLASSES = len(vocab_cuisine)

    date = datetime.now().strftime("%m-%d-%H-%M")
    RESULTS_FOLDER = os.path.join(os.getcwd(),"results",f"nn_{date}{balanced*'_bal'}_{file}")
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    train_loader, dev_loader, test_loader, weights_classes = createDataLoaders(df, vocab_ingrs,vocab_cuisine, balanced)

    net = Net(INPUT_SIZE, EMBED_DIM1, EMBED_DIM2, NUM_CLASSES).to(argv[0])

    loss, epoch, epoch_accuracy, epoch_test_accuracy, optimizer = train(net, train_loader, test_loader, vocab_ingrs, load, weights_classes)

    _, dev_fscore = test_score(net, dev_loader, vocab_ingrs, test=True,threshold=threshold)

    plotAccuracy(RESULTS_FOLDER, epoch_accuracy,epoch_test_accuracy)

    saveResults(RESULTS_FOLDER, net, loss, epoch, epoch_accuracy, epoch_test_accuracy, dev_fscore, optimizer)

if __name__=="__main__":
    file = FILES[2]
    balanced = False
    load = False
    main(sys.argv[1:],file, balanced, load)