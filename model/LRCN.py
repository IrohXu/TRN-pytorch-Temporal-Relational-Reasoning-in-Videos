import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
from .resnet import resnet50

class LRCNs(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, out_dim, device):
        super(LRCNs, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.device = device
        cell_list = []

        for i in range(0, self.num_layers):
            tmp_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(nn.LSTMCell(tmp_input_dim, hidden_dim))
        
        self.cell_list = nn.ModuleList(cell_list)
        self.backbone = resnet50(pretrained=True)
        self.fc = nn.Linear(self.hidden_dim, self.out_dim)

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append([torch.zeros(batch_size, self.hidden_dim),torch.zeros(batch_size, self.hidden_dim)])
        return init_states  
    
    def forward(self, x, hidden_state=None):
        hidden_state = self._init_hidden(batch_size=x.size(0))
        seq_len = x.size(1)
        cur_layer_input = x

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx][0].to(self.device), hidden_state[layer_idx][1].to(self.device)
            output_inner = []
            for t in range(seq_len):
                if layer_idx == 0:
                    cnn_feature=torch.squeeze(self.backbone(cur_layer_input[:, t, :, :, :]))
                    h, c = self.cell_list[layer_idx](cnn_feature, (h, c))
                else:
                    h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :], (h, c))

                if self.num_layers==layer_idx+1:
                    output_inner.append(self.fc(h))
                else:
                    output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

        output = torch.mean(layer_output, dim = 1)
        return output

if __name__ == "__main__":
    input_var = Variable(torch.randn(5, 8, 3, 224, 224))
    model = LRCNs(2048, 128, 2, 27)
    output = model(input_var)
    print(output.shape)
        
        



