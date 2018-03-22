from torch.autograd import Variable
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

n_predictions = 100

def main():
    toy_problem = ToyProblem()
    toy_prediction_model = ToyPredictionModel2()

    train(toy_problem, toy_prediction_model)

    test(toy_problem, toy_prediction_model)


class ToyProblem:
    def __init__(self):
        self.weight = 10
        self.bias = 2
        # self.loss_function = torch.nn.MSELoss()

    def loss_function(self, prediction, target):
        return -torch.log(torch.exp(-torch.pow(prediction - target, 2)).sum(dim=1) + 1e-5).sum()
        #return -torch.log(torch.exp(-torch.pow(prediction - target, 2)).sum(dim=1)).sum()
        #return torch.pow(torch.abs(prediction - target.expand(prediction.size())), 0.5).sum()


    def mean(self, data):
        return data*self.weight + self.bias

    def loss(self, target, prediction):
        return self.loss_function(prediction, target)

    @staticmethod
    def generate_data(n_samples):
        return torch.FloatTensor(n_samples, 1).normal_(0, 1)

    def generate_target(self, data):
        target = self.mean(data)
        displacement = torch.round(torch.FloatTensor(data.size()).normal_(0, 1))*10
        target += torch.FloatTensor(data.size()).normal_(0, 1) + displacement
        return target


class ToyPredictionModel(nn.Module):
    def __init__(self):
        super(ToyPredictionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)


class ToyPredictionModel2(nn.Module):
    def __init__(self):
        super(ToyPredictionModel2, self).__init__()
        self.layer1 = nn.Linear(1, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3_mu = nn.Linear(5, 5)
        self.layer3_sigma = nn.Linear(5, 5)
        self.layer4 = nn.Linear(5, 5)
        self.layer5 = nn.Linear(5, 5)
        self.layer6 = nn.Linear(5, 5)
        self.layer7 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        mu = self.layer3_mu(x)
        mu = (mu - mu.mean(dim=1, keepdim=True)) / (mu.std(dim=1, keepdim=True) + 1e-5)
        logvar = self.layer3_sigma(x)
        std = logvar.mul(0.5).exp_()
        x = mu + std*Variable(torch.FloatTensor(std.size()).normal_(0, 1))
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.layer5(x)
        x = F.relu(x)
        x = self.layer6(x)
        x = F.relu(x)
        x = self.layer7(x)
        return x, logvar.sum()


def train(toy_problem, toy_prediction_model, batch_size=500, n_iterations=10000):

    optimizer = optim.SGD(toy_prediction_model.parameters(), lr=1e-4, momentum=1e-5)
    toy_prediction_model.train()
    for iteration in range(n_iterations):
        data = toy_problem.generate_data(batch_size)
        target = toy_problem.generate_target(data)
        repeated_data = data.expand((-1, 5)).contiguous().view(-1, 1)

        repeated_data, target = Variable(repeated_data), Variable(target)

        predictions, logvars = toy_prediction_model(repeated_data)

        predictions = predictions.view((-1, 5))

        loss = toy_problem.loss(target, predictions) #- 1e-6*logvars.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 2 == 0:
            print('Iteration: {}\tLoss: {:.6f}'.format(
                iteration, torch.mean(loss.data)))
            # print([x for x in toy_prediction_model.named_parameters()])


def test(toy_problem, toy_prediction_model):

    n_samples = 100
    data = toy_problem.generate_data(n_samples)
    data, indices = data.sort(dim=0)

    target = toy_problem.generate_target(data)
    data, target = Variable(data), Variable(target)
    predictions = []
    for _ in range(n_predictions):
        prediction, logvar = toy_prediction_model(data)
        predictions.append(prediction)
    predictions = torch.cat(predictions, dim=1)

    plt.scatter(data.data.numpy(), target.data.numpy())
    for i in range(n_predictions):
        plt.scatter(data.data.numpy(), predictions.data.numpy()[:, i], c='r', alpha=1/n_predictions)
    plt.show()


if __name__ == '__main__':
    main()
