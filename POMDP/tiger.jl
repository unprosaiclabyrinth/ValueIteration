#
# Offline planning for the Tiger POMDP
#
@enum State TigerLeft TigerRight

@enum Action OpenLeft OpenRight Listen

@enum Observation GrowlLeft GrowlRight

function transition_p(s::State, a::Action, t::State)::Float64
    if a == Listen
        if s == t 1.0 else 0.0 end
    else
        0.5
    end
end

function observation_p(a::Action, t::State, o::Observation)::Float64
    if a == Listen
        if (t == TigerLeft && o == GrowlLeft) || (t == TigerRight && o == GrowlRight)
            0.85
        else
            0.15
        end
    else
        0.5
    end
end

function reward_f(s::State, a::Action)::Float64
    if a == Listen
        -1.0
    elseif (s == TigerLeft && a == OpenRight) || (s == TigerRight && a == OpenLeft)
        10.0
    else
        -100.0
    end
end

struct BeliefState
    ptl::Float64 # Probability of being in state TigerLeft
end

belief_distribution(b::BeliefState) = Dict(TigerLeft => b.ptl, TigerRight => 1 - b.ptl)

function belief_update(b::BeliefState, a::Action, o::Observation)::BeliefState
    prior = belief_distribution(b)
    posterior = Dict{State, Float64}(s => 0.0 for s in State)
    for s in State
        for t in State
            posterior[t] += prior[s] * transition_p(s, a, t) * observation_p(a, t, o)
        end
    end
    normalizer = sum(values(posterior)) # normalization constant
    BeliefState(posterior[TigerLeft] / normalizer)
end

function belief_observation_p(b::BeliefState, a::Action, o::Observation)::Float64
    prior = belief_distribution(b)
    po = 0.0
    for s in State
        for t in State
            po += prior[s] * transition_p(s, a, t) * observation_p(a, t, o)
        end
    end
    po # nothing but the normalization constant in the belief update
end

belief_reward_f(b::BeliefState, a::Action)::Float64 =
    (b.ptl * reward_f(TigerLeft, a)) + ((1 - b.ptl) * reward_f(TigerRight, a))
