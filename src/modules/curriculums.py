import isaaclab.envs.mdp as mdp


def ramp_force_amplitude(env, env_ids, data, initial, final, warmup_steps, ramp_steps):
    """Linearly ramp force amplitude from initial to final after a warmup period.

    Schedule:
        steps < warmup_steps                       → initial (e.g. 0 N)
        warmup_steps <= steps < warmup + ramp       → linear interpolation
        steps >= warmup + ramp                      → final  (e.g. 70 N)
    """
    step = env.common_step_counter
    if step < warmup_steps:
        target = initial
    elif step < warmup_steps + ramp_steps:
        progress = (step - warmup_steps) / ramp_steps
        target = initial + (final - initial) * progress
    else:
        target = final

    new_val = [target]
    if data != new_val:
        return new_val
    return mdp.modify_env_param.NO_CHANGE
