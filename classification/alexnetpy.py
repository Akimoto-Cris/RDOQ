import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    with torch.no_grad():
        weight = model.classifier[1].weight
        biases = model.classifier[1].bias
        conv2d = nn.Conv2d(256, 4096, kernel_size=6)
        conv2d.weight[:] = weight.reshape([4096,256,6,6])
        conv2d.bias[:] = biases[:]
        model.classifier[1] = conv2d
        
        weight = model.classifier[4].weight
        biases = model.classifier[4].bias
        conv2d = nn.Conv2d(4096, 4096, kernel_size=1)
        conv2d.weight[:] = weight.reshape([4096,4096,1,1])
        conv2d.bias[:] = biases[:]
        model.classifier[4] = conv2d
        
        weight = model.classifier[6].weight
        biases = model.classifier[6].bias
        conv2d = nn.Conv2d(4096, 1000, kernel_size=1)
        conv2d.weight[:] = weight.reshape([1000,4096,1,1])
        conv2d.bias[:] = biases[:]
        model.classifier[6] = conv2d
    return model
