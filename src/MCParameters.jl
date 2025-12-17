"""
    MoveWisdom

Structure to hold parameters and statistics for a single type of Monte Carlo move.
"""
@kwdef struct MoveWisdom
    step_size::Float64 = 0.2
    target_acceptance::Float64 = 0.5
    adaptation_rate::Float64 = 0.1
    min_step_size::Float64 = 1e-4
    max_step_size::Float64 = 1.0
    acceptance::Float64 = 0.0
    attempts::Int = 0
end

"""
    update_step_size!(wisdom::MoveWisdom)

Update the step size based on current acceptance rate.
"""
function update_step_size!(wisdom::MoveWisdom)
    ratio = wisdom.acceptance / wisdom.target_acceptance
    new_size = wisdom.step_size * (1 + wisdom.adaptation_rate * (ratio - 1))
    wisdom.step_size = clamp(new_size, wisdom.min_step_size, wisdom.max_step_size)
end

"""
    update_acceptance!(wisdom::MoveWisdom, accepted::Bool)

Update the acceptance statistics.
"""
function update_acceptance!(wisdom::MoveWisdom, accepted::Bool)
    wisdom.attempts += 1
    wisdom.acceptance = (wisdom.acceptance * (wisdom.attempts - 1) + accepted) / wisdom.attempts
end

"""
    reset!(wisdom::MoveWisdom)

Reset statistics while keeping parameters.
"""
function reset!(wisdom::MoveWisdom)
    wisdom.acceptance = 0.0
    wisdom.attempts = 0
end 