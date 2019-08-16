import torchvision
from torchvision.models import resnet18
import torch
from torch import nn

GPU = 2

model = resnet18()
#modify resnet to fit mnist data
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1,)

if __name__ == '__main__':


    if GPU is not None:
        model = model.cuda(GPU)

    batch_size_train = 16
    batch_size_test = 1000
    lr = 0.001
    log_interval = 1000
    n_epochs = 10
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=False,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=False,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    train_losses = []
    test_losses =[]
    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if GPU is not None:
                data = data.cuda(GPU)
                target = target.cuda(GPU)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                                epoch, batch_idx * len(data), len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if GPU is not None:
                    data = data.cuda(GPU)
                    target = target.cuda(GPU)
                output = model(data)
                test_loss += torch.nn.CrossEntropyLoss()(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()


    torch.save(model.state_dict(), 'resnet_model.ph')
