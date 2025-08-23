import torch
import copy

class EMA:
    def __init__(self, model, decay=0.995, device=None):
        self.decay = decay
        self.ema_model = self._copy_model(model)
        if device is not None:
            self.ema_model.to(device)
        self.ema_model.eval()

    def _copy_model(self, model):
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema_model.state_dict()
            for k in esd.keys():
                if k in msd:
                    if msd[k].dtype.is_floating_point:
                        esd[k].copy_(self.decay * esd[k] + (1. - self.decay) * msd[k])
                    else:
                        esd[k].copy_(msd[k])  # for buffers like batchnorm stats

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    
