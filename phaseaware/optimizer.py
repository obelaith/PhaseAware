import math
import torch
from torch.optim import Optimizer


class PhaseAwareOptimizer(Optimizer):
    """
    Phase-Aware Optimizer

    Combines:
    - Adam-style adaptive moments
    - phase-controlled learning rate scheduling
    - adaptive exploration noise
    - gradient-variance phase estimation
    - decoupled weight decay

    The phase variable φ ∈ [0,1] controls the transition
    from exploration to exploitation.
    """

    def __init__(
        self,
        params,
        lr_max=1e-3,
        lr_min=1e-4,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        noise_max=0.03,
        noise_clip=3.0,
        total_steps=10000,
        warmup_ratio=0.1,
        phase_type="time",
        var_window=50,
        time_weight=0.7,
        weight_decay=0.0,
        max_grad_norm=1.0,
    ):

        defaults = dict(
            lr_max=lr_max,
            lr_min=lr_min,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            noise_max=noise_max,
            noise_clip=noise_clip,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            phase_type=phase_type,
            var_window=var_window,
            time_weight=time_weight,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )

        super().__init__(params, defaults)

        self.step_count = 0

        self.phase = 0.0
        self.current_lr = 0.0
        self.current_noise = 0.0

        self.grad_history = []
        self.initial_variance = None


    def _time_phase(self):
        return min(
            1.0,
            self.step_count / max(self.defaults["total_steps"], 1)
        )


    def _compute_phase(self, grad_norm):

        phi_time = self._time_phase()

        if self.defaults["phase_type"] == "time":
            return phi_time


        window = self.defaults["var_window"]

        self.grad_history.append(float(grad_norm))

        if len(self.grad_history) > window * 3:
            self.grad_history = self.grad_history[-window * 2:]


        if len(self.grad_history) < window:
            return phi_time


        recent = self.grad_history[-window:]

        mean = sum(recent) / window

        variance = sum(
            (x - mean) ** 2 for x in recent
        ) / window


        if self.initial_variance is None:
            self.initial_variance = max(variance, 1e-12)


        variance_ratio = min(
            1.0,
            variance / self.initial_variance
        )


        stability_phase = 1.0 - variance_ratio

        tw = self.defaults["time_weight"]

        phi = (
            tw * phi_time +
            (1 - tw) * stability_phase
        )

        return max(
            phi_time,
            min(phi, 1.0)
        )


    def _compute_lr(self, phi):

        lr_max = self.defaults["lr_max"]
        lr_min = self.defaults["lr_min"]

        warmup = self.defaults["warmup_ratio"]


        if phi < warmup:
            return lr_max * phi / max(warmup, 1e-8)


        progress = (
            phi - warmup
        ) / max(1.0 - warmup, 1e-8)


        cosine = (
            1 + math.cos(math.pi * progress)
        ) / 2


        return (
            lr_min +
            (lr_max - lr_min) * cosine
        )


    def _sample_cauchy(self, shape, device):

        noise = torch.rand(
            shape,
            device=device
        ) - 0.5

        noise = torch.tan(
            math.pi * noise
        )

        return torch.clamp(
            noise,
            -self.defaults["noise_clip"],
            self.defaults["noise_clip"]
        )


    def _global_grad_norm(self):

        grads = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grads.append(
                        p.grad.detach().norm(2)
                    )

        if not grads:
            return torch.tensor(0.0)

        return torch.norm(
            torch.stack(grads)
        )


    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()


        self.step_count += 1


        grad_norm = self._global_grad_norm()

        if self.defaults["max_grad_norm"] > 0:

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ],
                self.defaults["max_grad_norm"]
            )


        phi = self._compute_phase(
            grad_norm.item()
        )


        lr = self._compute_lr(phi)

        self.phase = phi
        self.current_lr = lr


        beta1 = self.defaults["beta1"]
        beta2 = self.defaults["beta2"]
        eps = self.defaults["eps"]

        weight_decay = self.defaults["weight_decay"]


        bias1 = 1 - beta1 ** self.step_count
        bias2 = 1 - beta2 ** self.step_count


        noise_total = 0.0
        noise_count = 0


        for group in self.param_groups:

            for p in group["params"]:

                if p.grad is None:
                    continue


                if weight_decay != 0:

                    p.mul_(
                        1 - lr * weight_decay
                    )


                grad = p.grad


                state = self.state[p]


                if len(state) == 0:

                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)


                m = state["m"]
                v = state["v"]


                m.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1
                )


                v.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2
                )


                m_hat = m / bias1
                v_hat = v / bias2


                update = (
                    m_hat /
                    (v_hat.sqrt() + eps)
                )


                p.add_(
                    update,
                    alpha=-lr
                )


                rms_update = (
                    update.norm()
                    /
                    math.sqrt(update.numel())
                )


                noise_scale = (
                    self.defaults["noise_max"]
                    *
                    rms_update.item()
                    *
                    (1 - phi)
                )


                if noise_scale > 1e-12:

                    perturbation = self._sample_cauchy(
                        p.shape,
                        p.device
                    )

                    p.add_(
                        perturbation,
                        alpha=lr * noise_scale
                    )


                noise_total += noise_scale
                noise_count += 1



        self.current_noise = (
            lr *
            noise_total /
            max(noise_count, 1)
        )


        return loss



    def get_current_state(self):

        return {
            "step": self.step_count,
            "phase": self.phase,
            "lr": self.current_lr,
            "noise": self.current_noise,
            "momentum": self.defaults["beta1"],
        }