import torch
import torch.nn.functional as F

def compute_rpc_losses(
    s, a, r, s_next, done,
    encoder, dynamics, policy, q_func, target_q,
    lambda_info, gamma
):
    # Encode current and next states
    z, logq_z, _, _ = encoder(s)
    z_next, logq_z_next, _, _ = encoder(s_next)

    # Predict next latent distribution
    prior_dist = dynamics(z, a)
    logp_z_next = prior_dist.log_prob(z_next).sum(dim=-1, keepdim=True)

    # Info cost (KL(q || m) in nats): log q(z_{t+1}|s_{t+1}) - log m(z_{t+1}|z_t,a_t)
    info_cost = logq_z_next - logp_z_next

    # Augmented reward r~ = r - λ * info_cost   (stopgrad on info for Q-target)
    r_tilde = r - lambda_info * info_cost.detach()

    # Q-target
    with torch.no_grad():
        a_next, _ = policy(z_next)     # action logits or continuous proto-actions
        q_next = target_q(s_next, a_next)
        target_value = r_tilde + gamma * (1 - done) * q_next

    q_pred = q_func(s, a)
    q_loss = F.mse_loss(q_pred, target_value)

    # Actor/encoder/dynamics objective: maximize Q and predictability (low info cost)
    policy_loss = -q_func(s, a).mean() + lambda_info * info_cost.mean()

    return policy_loss, q_loss, info_cost.mean()
