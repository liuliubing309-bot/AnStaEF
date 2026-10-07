import torch
import torch.nn as nn
import torch.nn.functional as F


class AnStaEFLoss(nn.Module):
    """Forecasting loss with routing and candidate-diversity regularization."""

    def __init__(
        self,
        lambda_mae=0.5,
        lambda_balance=0.01,
        lambda_ortho=0.01,
        lambda_z=0.001,
        lambda_entropy=0.001,
    ):
        super(AnStaEFLoss, self).__init__()
        self.lambda_mae = lambda_mae
        self.lambda_balance = lambda_balance
        self.lambda_ortho = lambda_ortho
        self.lambda_z = lambda_z
        self.lambda_entropy = lambda_entropy
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
        self, pred, true, router_logits=None, candidate_pool=None, aux_scale=1.0
    ):
        loss_task = self.mse_loss(pred, true) + self.lambda_mae * self.l1_loss(
            pred, true
        )

        loss_balance = torch.tensor(0.0, device=pred.device)
        loss_ortho = torch.tensor(0.0, device=pred.device)
        loss_z = torch.tensor(0.0, device=pred.device)
        loss_entropy = torch.tensor(0.0, device=pred.device)

        if router_logits is not None:
            loss_balance = self._compute_balance_loss(router_logits)
            loss_z = self._compute_z_loss(router_logits)
            loss_entropy = self._compute_entropy_loss(router_logits)

        if candidate_pool is not None:
            loss_ortho = self._compute_orthogonality_loss(candidate_pool)

        total_loss = (
            loss_task
            + (aux_scale * self.lambda_balance * loss_balance)
            + (aux_scale * self.lambda_ortho * loss_ortho)
            + (aux_scale * self.lambda_z * loss_z)
            + (aux_scale * self.lambda_entropy * loss_entropy)
        )

        return total_loss, loss_task, loss_balance, loss_ortho, loss_z, loss_entropy

    def _compute_balance_loss(self, router_logits):
        logits = router_logits.reshape(
            -1, router_logits.size(-1)
        )  # (N_samples, N_experts)

        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)

        _, top1_indices = torch.max(logits, dim=-1)
        mask = F.one_hot(top1_indices, num_classes=logits.size(-1)).float()
        fraction = mask.mean(dim=0)

        num_experts = logits.size(-1)

        balance_loss = num_experts * torch.sum(mean_probs * fraction)

        return balance_loss

    def _compute_z_loss(self, router_logits):
        logits = router_logits.reshape(-1, router_logits.size(-1))
        z = torch.logsumexp(logits, dim=-1)
        return torch.mean(z**2)

    def _compute_entropy_loss(self, router_logits):
        logits = router_logits.reshape(-1, router_logits.size(-1))
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        return -torch.mean(entropy)

    def _compute_orthogonality_loss(self, candidate_pool):
        outputs = candidate_pool.reshape(
            -1, candidate_pool.size(2), candidate_pool.size(3)
        )
        outputs_norm = F.normalize(outputs, p=2, dim=-1, eps=1e-8)
        cosine_sim_matrix = torch.bmm(outputs_norm, outputs_norm.transpose(1, 2))
        num_experts = outputs.size(1)
        identity = torch.eye(num_experts, device=outputs.device).unsqueeze(0)
        ortho_loss = torch.mean(
            torch.norm(cosine_sim_matrix - identity, p='fro', dim=(1, 2)) ** 2
        )
        return ortho_loss
