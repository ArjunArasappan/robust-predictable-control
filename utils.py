def soft_update(target, source, tau=0.005):
    for target_param, src_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + src_param.data * tau)
