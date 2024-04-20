"""
Simple training loop for pytorch modules
"""
import torch
import torch.optim as optim
import torch.nn.functional as F


def torchTrain(model,
               train_dataset,
               test_dataset,
               save_path,
               n_epochs=3,
               batch_size_train=64,
               batch_size_test=100,
               learning_rate=0.01,
               momentum=0.5,
               log_interval=10,
               device=torch.device("cpu")):
    """
    Simple training loop for MNIST dataset, saves optimal state
    :param model: model to be trained
    :param save_path: name of file to save model to
    :param n_epochs: number of epochs to train for
    :param batch_size_train: training batch size
    :param batch_size_test: testing batch size
    :param learning_rate: learning rate for SGD
    :param momentum: momentum for SGD
    :param log_interval: interval between logging training results
    :param device: device to train on
    :return: None
    """

    model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = F.nll_loss(torch.log(output), target.to(device))
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_dataset)))

    def val(previous_loss):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data.to(device))
                test_loss += F.nll_loss(torch.log(output), target.to(device), size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.cpu().eq(target.data.view_as(pred.cpu())).sum()
        test_loss /= len(test_dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_dataset),
            100. * correct / len(test_dataset)))
        if test_loss < previous_loss:
            print("saving")
            torch.save(model.state_dict(), f'./states/{save_path}.pth')

    for epoch in range(1, n_epochs + 1):
        train(epoch)
        previous_loss = float('inf') if epoch == 1 else test_losses[epoch-2]
        val(previous_loss)
