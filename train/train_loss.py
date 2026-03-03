import torch 
from typing import Optional

class SurrogateLoss(torch.nn.Module):
    def __init__(self, reduced_weight_list: list):
        super().__init__()
        self.reduced_weight_list = reduced_weight_list

    def forward(self, y: torch.Tensor, sample_index: int, sub_dim: Optional[int] = None):
        if sub_dim is None:
            sub_dim = y.shape[0]
        assert sub_dim <= y.shape[0]

        reduced_weight = self.reduced_weight_list[sample_index]
        sub_y = y[:sub_dim]

        quadratic_loss = torch.dot(sub_y, reduced_weight['quadratic'][:sub_dim, :sub_dim] @ sub_y)
        linear_loss = 2.0*torch.dot(sub_y, reduced_weight['linear'][:sub_dim]) 

        bias_loss = reduced_weight['bias']
        loss = quadratic_loss + linear_loss + bias_loss
        
        return loss

    def to(self, device):
        for i in range(len(self.reduced_weight_list)):
            self.reduced_weight_list[i]['quadratic'] = self.reduced_weight_list[i]['quadratic'].to(device)
            self.reduced_weight_list[i]['linear'] = self.reduced_weight_list[i]['linear'].to(device)
            self.reduced_weight_list[i]['bias'] = self.reduced_weight_list[i]['bias'].to(device)

        return self
    
    def astype(self, dtype):
        for i in range(len(self.reduced_weight_list)):
            self.reduced_weight_list[i]['quadratic'] = self.reduced_weight_list[i]['quadratic'].to(dtype)
            self.reduced_weight_list[i]['linear'] = self.reduced_weight_list[i]['linear'].to(dtype)
            self.reduced_weight_list[i]['bias'] = self.reduced_weight_list[i]['bias'].to(dtype)
        return self
