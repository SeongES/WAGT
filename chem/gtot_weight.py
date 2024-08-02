import numpy as np
import torch
import torch.nn as nn
import torch_geometric.utils as PyG_utils
import functools
import time

from collections import OrderedDict, deque


# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# Adapted from https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py
class GTOT(nn.Module):
    r"""
        GTOT implementation.
    """

    def __init__(self, eps=0.1, thresh=0.1, max_iter=100, reduction='none'):
        super(GTOT, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.thresh = thresh
        self.mask_matrix = None

    def marginal_prob_unform(self, N_s=None, N_t=None, mask=None, ):
        if mask is not None:
            mask = mask.float()
            # uniform distribution
            mask_mean = (1 / mask.sum(1)).unsqueeze(1)
            mu = mask * mask_mean  # 1/n
            # mu = mu.unsqueeze(2)
        else:
            mu = torch.ones(self.bs, N_s) / N_s
        nu = mu.clone().detach()
        return mu, nu

    def forward(self, x, y, C=None, A=None, mask=None):
        # The Sinkhorn algorithm takes as input three variables :
        if C is None:
            C = self._cost_matrix(x, y)  # Wasserstein cost function
            C = C / C.max()
        if A is not None:
            if A.type().startswith('torch.cuda.sparse'):
                self.sparse = True
                C = A.to_dense() * C
            else:
                self.sparse = False
                C = A * C
        N_s = x.shape[-2]
        N_t = y.shape[-2]
        if x.dim() == 2:
            self.bs = 1
        else:
            self.bs = x.shape[0]
        # both marginals are fixed with equal weights
        if mask is None:
            mu = torch.empty(self.bs, N_s, dtype=torch.float, device=C.device,
                             requires_grad=False).fill_(1.0 / N_s).squeeze()
            nu = torch.empty(self.bs, N_t, dtype=torch.float, device=C.device,
                             requires_grad=False).fill_(1.0 / N_t).squeeze()
        else:
            mu, nu = self.marginal_prob_unform(N_s=N_s, N_t=N_t, mask=mask)

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = self.thresh
        # Sinkhorn iterations
        for i in range(self.max_iter):
            # print('i', i)
            u1 = u  # useful to check the update

            if mask is None:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
            else:
                u = self.eps * (torch.log(mu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A), dim=-1)) + u
                u = mask * u
                v = self.eps * (
                        torch.log(nu + 1e-8) - self.log_sum(self.exp_M(C, u, v, A=A).transpose(-2, -1), dim=-1)) + v
                v = mask * v

            err = (u - u1).abs().sum(-1).max()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v

        pi = self.exp_M(C, U, V, A=A)
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        if torch.isnan(cost.sum()):
            raise
        return cost, pi, C

    def M(self, C, u, v, A=None):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"


        S = (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
        return S

    def exp_M(self, C, u, v, A=None):
        if A is not None:
            if self.sparse:
                a = A.to_dense()
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-a).to(torch.bool),value=0)
            else:
                S = torch.exp(self.M(C, u, v)).masked_fill(mask = (1-A).to(torch.bool),value=0)

            return S
        elif self.mask_matrix is not None:
            return self.mask_matrix * torch.exp(self.M(C, u, v))
        else:
            return torch.exp(self.M(C, u, v))

    def log_sum(self, input_tensor, dim=-1, mask=None):
        s = torch.sum(input_tensor, dim=dim)
        out = torch.log(1e-8 + s)
        if torch.isnan(out.sum()):
            raise
        if mask is not None:
            out = mask * out
        return out

    def cost_matrix_batch_torch(self, x, y, mask=None):
        "Returns the cosine distance batchwise"
        # x is the source feature: bs * d * m
        # y is the target feature: bs * d * m
        # return: bs * n * m
        # print(x.size())
        bs = list(x.size())[0]
        D = x.size(1)
        assert (x.size(1) == y.size(1))
        x = x.contiguous().view(bs, D, -1)  # bs * d * m
        x = x.div(torch.norm(x, p=2, dim=1, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=1, keepdim=True) + 1e-12)
        cos_dis = torch.bmm(torch.transpose(x, 1, 2), y)  # .transpose(1,2)
        cos_dis = 1 - cos_dis  # to minimize this value
        # cos_dis = - cos_dis
        if mask is not None:
            mask0 = mask.unsqueeze(2).clone().float()
            self.mask_matrix = torch.bmm(mask0, (mask0.transpose(2, 1)))  # torch.ones_like(C)
            cos_dis = cos_dis * self.mask_matrix
        if torch.isnan(cos_dis.sum()):
            raise
        return cos_dis.transpose(2, 1)
    

    def cost_matrix_torch(self, x, y):
        "Returns the cosine distance"
        # x is the image embedding
        # y is the text embedding
        D = x.size(0)
        x = x.view(D, -1)
        assert (x.size(0) == y.size(0))
        x = x.div(torch.norm(x, p=2, dim=0, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=0, keepdim=True) + 1e-12)
        cos_dis = torch.mm(torch.transpose(y, 0, 1), x)  # .t()
        cos_dis = 1 - cos_dis  # to minimize this value
        return cos_dis

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    

    def model_norm(self, model):
        total_sum = 0.0
        total_values = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and 'graph_pred_linear' not in name: 
                total_sum += param.data.sum() 
                total_values += param.data.numel() 

        # Calculating the average
        average = total_sum / total_values
        return average
    
    def cost_matrix_from_weight(self, model, A, x, y, mask=None):  
        weight_norm = self.model_norm(model)
        adj_size = A.size(1)
        notdiag_one = torch.ones((adj_size, adj_size), dtype=torch.int, device=A.device)#*0.01
        notdiag_one.fill_diagonal_(0)
        cost_dist = weight_norm + notdiag_one

        cost_dist = cost_dist.repeat(x.size(0),1,1)

        if mask is not None:
            mask0 = mask.unsqueeze(2).clone().float()
            self.mask_matrix = torch.bmm(mask0, (mask0.transpose(2, 1)))  # torch.ones_like(C)
            cost_dist = cost_dist * self.mask_matrix
        if torch.isnan(cost_dist.sum()):
            raise

        return cost_dist

class GTOTRegularization(nn.Module):
    r"""
       GTOT regularization for finetuning
    Shape:
        - Output: scalar.

    """

    def __init__(self, order=1, args=None):
        super(GTOTRegularization, self).__init__()
        self.Gtot = GTOT(eps=0.1, thresh=0.1, max_iter=100, reduction=None)

        self.args = args
        self.order = order
        self.M = 0.05

    def sensible_normalize(self, C, mask=None):
        d_max = torch.max(C.abs().view(C.shape[0], -1), -1)[0]
        d_max = d_max.unsqueeze(1).unsqueeze(2)
        d_max[d_max == 0] = 1e9
        C = (C / d_max)
        if torch.isnan(C.sum()):
            raise
        return C


    
    def got_dist(self, model,f_s, f_t, A=None, norm='sqrt', mask=None):
        ## cosine distance
        '''if there is batch graph, the mask should be added to the cos_distance to make the dist to be zeros when the vector is padding.'''
        cos_distance = self.Gtot.cost_matrix_from_weight(model, A, f_s.transpose(2, 1), f_t.transpose(2, 1), mask=mask)

        ## D= max(D_{cos},threshold)
        threshod = False
        penalty = 50
        if threshod:
            beta = 0.1
            min_score = cos_distance.min()
            max_score = cos_distance.max()
            threshold = min_score + beta * (max_score - min_score)
            cos_dist = torch.nn.functional.relu(cos_distance - threshold)
        else:
            cos_dist = cos_distance

            self.sensible_normalize(cos_dist, mask=mask)


        # use different A^{order} as mask matrix
        if self.order == 0:
            A = torch.stack([torch.diag(mask_i.type_as(A)) for mask_i in mask])
        elif self.order == 1:
            A = A

        elif self.order >= 9:
            A = self.Gtot.mask_matrix
        elif self.order > 0:  # order =1
            A0 = A
            for i in range(self.order - 1):
                A = A.bmm(A0)
            A = torch.sign(A)
        else:
            raise

        if A is not None:
            A = self.Gtot.mask_matrix * A
            ## find the isolated Points in A , which will make the row (and symmetrical colum )of the cost matrix
            # become zero. This causes numerical overflow
            row, col = torch.nonzero((A.sum(-1) + (~mask)) == 0, as_tuple=True)
            mask[row, col] = False
        ## Masked OT with A^{order} as mask matrix
        wd, P, C = self.Gtot(x=f_s, y=f_t, A=A, C=cos_dist, mask=mask)
        twd = .5 * torch.mean(wd)

        return twd

    def forward(self,model,  layer_outputs_source, layer_outputs_target, *argv):
        '''
        Args:
            layer_outputs_source:
            layer_outputs_target:
            batch: batch is a column vector which maps each node to its respective graph in the batch

        Returns:

        '''

        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")
        output = 0.0

        for i, (fm_src, fm_tgt) in enumerate(zip(layer_outputs_source.values(), layer_outputs_target.values())):


            b_nodes_fea_s, b_mask_s = PyG_utils.to_dense_batch(x=fm_src.detach(), batch=batch)
            b_nodes_fea_t, b_mask_t = PyG_utils.to_dense_batch(x=fm_tgt, batch=batch)

            edge_index, edge_weight = PyG_utils.add_remaining_self_loops(edge_index, num_nodes=fm_tgt.size(0))
            b_A = PyG_utils.to_dense_adj(edge_index, batch=batch)
            ##  GTOT distance
            distance = self.got_dist(model=model, f_s=b_nodes_fea_s.detach(), f_t=b_nodes_fea_t, A=b_A, mask=b_mask_t)


            output = output + torch.sum(distance)

        return output
    
def get_attribute(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


class IntermediateLayerGetter(object):
    r"""
    Wraps a model to get intermediate output values of selected layers.

    Args:
       model (torch.nn.Module): The model to collect intermediate layer feature maps.
       return_layers (list): The names of selected modules to return the output.
       keep_output (bool): If True, `model_output` contains the final model's output, else return None. Default: True

    Returns:
       - An OrderedDict of intermediate outputs. The keys are selected layer names in `return_layers` and the values are the feature map outputs. The order is the same as `return_layers`.
       - The model's final output. If `keep_output` is False, return None.

    """

    def __init__(self, model, return_layers, keep_output=True):
        self._model = model
        self.return_layers = return_layers
        self.keep_output = keep_output

    def __call__(self, *args, **kwargs):
        ret = OrderedDict()
        handles = []
        for name in self.return_layers:
            layer = get_attribute(self._model, name)

            def hook(module, input, output, name=name):
                ret[name] = output

            try:
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')
            handles.append(h)
        # todo
        if self.keep_output:
            output = self._model(*args, **kwargs)
        else:
            self._model(*args, **kwargs)
            output = None

        for h in handles:
            h.remove()

        return ret, output