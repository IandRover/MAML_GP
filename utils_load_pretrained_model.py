import torch.nn.functional as F
import torch
torch.manual_seed(1)

from torchvision.models import resnet50, ResNet50_Weights
m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.pool1 = torch.nn.MaxPool2d(3, 3)

    def forward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x.transpose(0,3,1,2))/5+0.5
            x = self.pool1(F.relu(self.conv1(x)))
            return x.permute(0,2,3,1).detach().cpu().numpy()

net = Net()
net.eval()

def enc(x): return net(x)
def enc0(x): return x