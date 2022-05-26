import torch
from torch.autograd import grad


class Weighter:
    def __init__(self, update_num, beta, alpha, method):
        super(Weighter, self).__init__()

        # initialize the alpha parameters
        self.aux_alpha = alpha if method != 'GCLB' else None

        # initialize the gradients of alpha parameters
        self.beta = beta
        self.aux_alpha_grad = 0

        self.step_count = 0
        self.update_num = update_num
        self.aux_model_grad = None
        self.method = method

    def accumulate(self, aux_loss, model):
        aux_model_grad = list(
            grad(aux_loss, model.parameters(), retain_graph=True, allow_unused=True))
        if self.aux_model_grad is None:
            self.aux_model_grad = aux_model_grad
        else:
            for i in range(len(self.aux_model_grad)):
                self.aux_model_grad[i] = self.aux_model_grad[i] + \
                    aux_model_grad[i]

    @ torch.no_grad()
    def update(self, model):
        main_model_grad = [
            param.grad.data for param in model.parameters() if param.grad is not None]
        proj_alpha, dot, ga_sqr = self.compute_proj(main_model_grad)

        if self.method == 'D4AM':
            alpha = proj_alpha + self.aux_alpha
        elif self.method == 'SRPR':
            alpha = self.aux_alpha
        elif self.method == 'GCLB':
            alpha = proj_alpha

        for gm, ga in zip(main_model_grad, self.aux_model_grad):
            gm.add_(alpha * ga)

        self.aux_model_grad = None
        if self.method in ['D4AM', 'SRPR']:
            self.update_aux_alpha(proj_alpha, dot, ga_sqr)

    @ torch.no_grad()
    def update_aux_alpha(self, proj_alpha, dot, ga_sqr):
        self.step_count += 1
        self.aux_alpha_grad += -2 * \
            (dot + proj_alpha * ga_sqr) + 2 * self.aux_alpha * ga_sqr
        if self.step_count == self.update_num:
            g = self.aux_alpha_grad / self.update_num
            alpha_step = max(-1, min(1.0, - self.beta * g))
            self.aux_alpha = max(0, self.aux_alpha + alpha_step)
            self.aux_alpha_grad = 0
            self.step_count = 0

    @ torch.no_grad()
    def compute_proj(self, main_model_grad):
        dot = 0
        ga_sqr = 0
        for gm, ga in zip(main_model_grad, self.aux_model_grad):
            dot += torch.sum(gm * ga).item()
            ga_sqr += ga.pow(2).sum().item()

        proj_alpha = - dot / ga_sqr if dot < 0 else 0
        assert proj_alpha >= 0
        return proj_alpha, dot, ga_sqr
