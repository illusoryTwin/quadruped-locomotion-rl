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


def staged_force_ramp(env, env_ids, data, stages, steps_per_stage):
    """Per-stage linear force ramp.

    Args:
        stages: list of (start_force, end_force) tuples per stage.
        steps_per_stage: number of steps per stage.
    """
    step = env.common_step_counter
    stage_idx = min(step // steps_per_stage, len(stages) - 1)
    stage_step = step - stage_idx * steps_per_stage
    start, end = stages[stage_idx]
    progress = min(stage_step / steps_per_stage, 1.0)
    target = start + (end - start) * progress

    new_val = [target]
    if data != new_val:
        return new_val
    return mdp.modify_env_param.NO_CHANGE


def multi_stage_stiffness(env, env_ids, data, stages, steps_per_stage):
    """Fix stiffness kp range to a single value per stage.

    Args:
        stages: list of stiffness values, e.g. [700, 600, 500, 400, 350].
        steps_per_stage: number of steps before switching to the next stiffness.
    """
    step = env.common_step_counter
    stage_idx = min(step // steps_per_stage, len(stages) - 1)
    target_kp = stages[stage_idx]

    new_val = (target_kp, target_kp)
    if data != new_val:
        return new_val
    return mdp.modify_env_param.NO_CHANGE
