import warnings
import torch
from piq import psnr, ssim, LPIPS

__EVALUATOR__ = {}


def register_evaluator(name: str):
    def wrapper(cls):
        if __EVALUATOR__.get(name, None):
            if __EVALUATOR__[name] != cls:
                warnings.warn(f"Name {name} is already registered!", UserWarning)
        __EVALUATOR__[name] = cls
        cls.name = name
        return cls

    return wrapper


def get_evaluator(name: str, **kwargs):
    if __EVALUATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __EVALUATOR__[name](**kwargs)


@register_evaluator('mri')
class MRIEvaluator:
    def __init__(self, op):
        self.main_eval_fn_name = 'psnr'
        self.op = op
    
    def norm(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def __call__(self, x, y, xhat):
        # compute psnr between x and xhat
        psnr_score = psnr(self.norm(x[0]), self.norm(xhat[0]), data_range=1.0)
        ssim_score = ssim(self.norm(x[0]), self.norm(xhat[0]), data_range=1.0)
        return {'psnr': psnr_score, 'ssim': ssim_score}
    


@register_evaluator('blackhole')
class BlackholeEvaluator:
    def __init__(self, op):
        self.main_eval_fn_name = 'chisq'
        self.op = op
    
    def __call__(self, x, y, xhat):
        results = self.op.evaluate_chisq(xhat, y, chi_sq_list=['cphase', 'logcamp'], normalize=True)
        # take the maximum value of chi-squared statistics
        metrics = torch.max(results['cphase'], results['logcamp'])
        return {self.main_eval_fn_name: metrics}
    
