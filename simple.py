from torch.autograd import Variable
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

n_predictions = 10

def main():
    toy_problem = ToyProblem()
    toy_prediction_model = ToyPredictionModel2()

    train(toy_problem, toy_prediction_model)

    test(toy_problem, toy_prediction_model)


class ToyProblem:
    def __init__(self):
        pass

    def loss_function(self, prediction, target):
        return -torch.exp(-torch.pow(prediction - target, 2)).sum(dim=1).prod()
        #return -torch.log(torch.exp(-torch.pow(prediction - target, 2)).sum(dim=1)).sum()
        #return torch.pow(torch.abs(prediction - target.expand(prediction.size())), 0.5).sum()

    def loss(self, target, prediction):
        return self.loss_function(prediction, target)

    def generate_target(self, data):
        target = self.mean(data)
        indicator = torch.FloatTensor(data.size()).bernoulli_(0.5)
        target += torch.FloatTensor(data.size()).normal_(-2, 1)*indicator
        target += torch.FloatTensor(data.size()).normal_(2, 1)*(1-indicator)
        return target


class ToyPredictionModel(nn.Module):
    def __init__(self):
        super(ToyPredictionModel, self).__init__()
        self.layer1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer1(x)


def train(toy_problem, toy_prediction_model, batch_size=500, n_iterations=10000):

    optimizer = optim.SGD(toy_prediction_model.parameters(), lr=1e-4, momentum=1e-5)
    toy_prediction_model.train()
    for iteration in range(n_iterations):
        data = toy_problem.generate_data(batch_size)
        target = toy_problem.generate_target(data)
        data, target = Variable(data), Variable(target)

        predictions = []
        logvars = []
        for _ in range(n_predictions):
            prediction, logvar = toy_prediction_model(data)
            predictions.append(prediction)
            logvars.append(logvar)
        predictions = torch.cat(predictions, dim=1)
        logvars = torch.cat(logvars)

        loss = toy_problem.loss(target, predictions) - 1e-4*logvars.sum()

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
        plt.scatter(data.data.numpy(), predictions.data.numpy()[:, i], c='r')
    plt.show()


if __name__ == '__main__':
    main()
