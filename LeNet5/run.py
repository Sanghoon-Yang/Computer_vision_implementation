from model import LeNet5
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import numpy as np

if __name__=='__main__':
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.manual_seed_all(1)
    else:
        device = 'cpu'
    lenet = LeNet5().to(device)

    batch_size = 256
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lenet.parameters(), lr=0.001)


    #load MNIST dataset
    train_data = datasets.mnist.MNIST('./', train=True, download=True, transform= transforms.ToTensor())
    test_data = datasets.mnist.MNIST('./', train=False, download=True, transform= transforms.ToTensor())

    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    #start training
    total_epochs = 100
    for epoch in range(total_epochs):
        lenet.train()
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = lenet(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

        print("epoch{}：완료\n".format(epoch))
    print('러닝 완료')
    all_correct_num = 0
    all_sample_num = 0
    #lenet evaluation
    lenet.eval()

    for x,y in test_loader:
        x = x.to(device)
        y = y.to(device)
        test_out = lenet(x)
        predict_y = torch.argmax(test_out, dim=-1)
        current_correct_num = predict_y == y
        all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
        all_sample_num += current_correct_num.shape[0]

    acc = all_correct_num / all_sample_num
    print('accuracy: {:.3f}'.format(acc), flush=True)

