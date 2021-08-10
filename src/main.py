from __future__ import print_function
from random import random
from random import seed
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as mplot
from Initialization import Initialization

def train(model, device, train_loader, optimizer, criterion):
    total_loss, correct = 0.0, 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        model = model.to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, target)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    total_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)

    return total_loss, accuracy

def test(model, device, test_loader, criterion):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            model = model.to(device)
            prediction = model(data)
            test_loss += criterion(prediction, target).item()
            pred = prediction.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, correct, accuracy

def weights_init_apply_with_bias(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias, 0.01)  # 0.01

def weights_init_apply_without_bias(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias, 0.00)

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        # torch.nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    
# add
def identical(input):
    return input

class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, input):
        return identical(input)
# add

class Net(nn.Module):
    def __init__(self, layer_unit_count_list, active_func_list, batch_size, retain_intermedium_grad = False):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.retain_intermedium_grad = retain_intermedium_grad
        self.fc = torch.nn.ModuleList()
        self.ac = torch.nn.ModuleList()
        self.intermedia_x = []
        self.intermedia_y = []
        for i in range(len(layer_unit_count_list) - 1):
            # add linear layer
            self.fc.append(torch.nn.Linear(layer_unit_count_list[i], layer_unit_count_list[i + 1]))
            # store output of linear layer
            self.intermedia_x.append(None)
            # add activation layer
            if (active_func_list[i] == 'tanh'):
                self.ac.append(torch.nn.Tanh())
            elif (active_func_list[i] == 'sigmoid'):
                self.ac.append(torch.nn.Sigmoid())
            elif (active_func_list[i] == 'relu'):
                self.ac.append(torch.nn.ReLU())            
            elif (active_func_list[i] == 'none'):
                self.ac.append(Identical())
            else:
                raise TypeError('activation type {} is out of range'.format(active_func_list[i]))
            # store output of activation layer
            self.intermedia_y.append(None)

    def forward(self, x):
        # reshape input
        layer_input = x.view(self.batch_size, -1)
        for i in range(len(self.fc)):
            if i > 0:
                layer_input = self.intermedia_y[i - 1]
            self.intermedia_x[i] = self.fc[i](layer_input)
            self.intermedia_y[i] = self.ac[i](self.intermedia_x[i])
            if self.retain_intermedium_grad:
                self.intermedia_x[i].retain_grad()
                self.intermedia_y[i].retain_grad()
        return self.intermedia_y[-1]


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='input learning rate for training (default: 0.2)')
    parser.add_argument('--training_epochs', type=int, default=101,
                        help='input training epochs for training (default: 501)')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--layer_unit_count_list',  nargs="*",  type=int,  default=[784, 100, 10])
    parser.add_argument('--active_func_list', nargs="*", type=str, default=["relu", "none"])
    parser.add_argument('--alphas',  nargs="*",  type=float,  default=[0, 1])
    
    parser.add_argument('--init_model_with_bias', type=int, default=0,
                        help='init model with bias as 0 or constant positive value')
    parser.add_argument('--random_seed', type=int, default=1, 
                        help='input random seed for training (default: 1)')   #  9
    parser.add_argument('--model_name', type=str, default= "test")
    parser.add_argument('--is_init', type=int, default=1)
    parser.add_argument('--candidate_weights', type=int, default=5)   #  9
    
    parser.add_argument('--init_num_sample', type=int, default=60000)
    parser.add_argument('--max_num_candidate', type=int, default=1000)
    # set common config
    args = parser.parse_args()
    assert len(args.layer_unit_count_list) - 1 == len(args.active_func_list)
    assert len(args.active_func_list) == len(args.alphas)
    print("alphas   ", args.alphas)
    # get result file name
    # Set random seed
    random_seed = args.random_seed
    seed(random_seed)  # python random seed
    torch.manual_seed(random_seed)  # pytorch random seed

    # set training config
    batch_size = args.batch_size
    training_epochs = args.training_epochs
    learning_rate = args.learning_rate

    # set cpu or gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    layer_unit_count_list = args.layer_unit_count_list  # [784, 256, 64, 10]
    active_func_list = args.active_func_list  # ["relu", "relu", "none"]
    init_model_with_bias = args.init_model_with_bias

    assert len(layer_unit_count_list) == len(active_func_list) + 1
    # # create NN layer by layer
    print(".......", layer_unit_count_list, active_func_list, batch_size)
    model = Net(layer_unit_count_list, active_func_list, batch_size)
    if not args.is_init:
       model.apply(weights_init)
    
    model.to(device)
    print(learning_rate)
    print(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # , momentum=0.9
    criterion = nn.CrossEntropyLoss()

    # # data set loader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    record_dict = {}
    if args.is_init:
        print(args.is_init)
        print("[...] Start initializing...")
        initializer = Initialization(trainset, args.init_num_sample, device, args, args.alphas)
        initializer.init_fc(model, args.max_num_candidate, init_model_with_bias > 0)
        print("[+] Finished intializing")
    


    # train and test the created model

    for epoch in range(0, training_epochs):
        if epoch == 0:
            print('Test initialization')
            test_loss, test_correct, test_accuracy = test(model, device, test_loader, criterion)
            print('Testing set: Average loss: {:.4f}, Accuracy: ({:.4f}%)'.format(test_loss, test_accuracy))
            print('Test initialization end')
            
            

        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion)
        print('Train_epoch\tTrain_loss\tTrain_acc')
        print('{}\t{:.6f}\t{:.4f}'.format(epoch, train_loss, train_accuracy))
        
        test_loss, test_correct, test_accuracy = test(model, device, test_loader, criterion)
             
        print('Test_epoch\tTest_loss\tTest_accuracy')
        print('{}\t{:.6f}\t{:.4f}'.format(epoch, test_loss, test_accuracy))




if __name__ == '__main__':
    main()
