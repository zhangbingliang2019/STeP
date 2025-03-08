from torch.autograd import grad
import torch
from forward_operator import Operator

class LatentWrapper(Operator):
    def __init__(self, op, model):
        super().__init__(sigma=op.sigma)
        self.op = op
        self.model = model

    def __call__(self, x):
        decoded = self.model.decode(x)
        return self.op(decoded)


    def loss(self, pred, observation):
        decoded = self.model.decode(pred)
        return self.op.loss(decoded.float(), observation)

    def gradient(self, pred, observation, return_loss=False):
        pred_tmp = pred.clone().detach().requires_grad_(True)
        loss = self.loss(pred_tmp, observation).sum()
        pred_grad = grad(loss, pred_tmp)[0]
        pred_grad = pred_grad.to(pred.dtype)
        # clip the gradient
        pred_grad = torch.clamp(pred_grad, -1, 1)
        if return_loss:
            return pred_grad, loss
        else:
            return pred_grad