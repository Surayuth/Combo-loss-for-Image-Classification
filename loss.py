class ComboLoss(nn.Module):
    def __init__(self, beta=0.6, num_classes=4):
        super(ComboLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, output, target):
        p = F.softmax(output, dim=1)
        t = F.one_hot(target, self.num_classes)
        L = -(self.beta * t * torch.log(p) + (1 - self.beta) * (1 - t) * torch.log(1 - p)).sum(dim=-1)
        return L.mean()
