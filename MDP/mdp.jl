using JSON3
using Printf

const State = Tuple{Int, Int}

abstract type Action end

struct North <: Action end
struct South <: Action end
struct East <: Action end
struct West <: Action end

function Base.show(io::IO, a::Action)
    if a isa North
        print(io, "↑")
    elseif a isa South
        print(io, "↓")
    elseif a isa East
        print(io, "→")
    else
        print(io, "←")
    end
end

rel_dir_to_action(::North) = Dict("forward" => North(), "left" => West(), "right" => East(), "back" => South())
rel_dir_to_action(::South) = Dict("forward" => South(), "left" => East(), "right" => West(), "back" => North())
rel_dir_to_action(::East) = Dict("forward" => East(), "left" => North(), "right" => South(), "back" => West())
rel_dir_to_action(::West) = Dict("forward" => West(), "left" => South(), "right" => North(), "back" => East())

struct GridWorldMDP
    nrows::Int
    ncols::Int
    blocked::Set{State}
    terminal_states::Set{State}
    terminal_rewards::Dict{State, Float64}
    nonterminal_reward::Float64
    transition_probabilities::Dict{String, Float64}
    discount::Float64
    epsilon::Float64
    states::Set{State}
    actions::Vector{Action}

    function GridWorldMDP(filename::String)
        mdp = JSON3.parsefile(filename)
        nrows = Int(mdp["size"]["rows"])
        ncols = Int(mdp["size"]["columns"])
        blocked = Set{State}(map(x -> (Int(x[1]), Int(x[2])), mdp["walls"]))
        terminal_states = Set{State}(map(x -> (Int(x[1]), Int(x[2])), mdp["terminal_states"]["positions"]))
        new(
            nrows,                                                                                       # nrows
            ncols,                                                                                       # ncols
            blocked,                                                                                     # blocked
            terminal_states,                                                                             # terminal_states
            Dict{State, Float64}(zip(terminal_states, map(Float64, mdp["terminal_states"]["rewards"]))), # terminal_rewards
            Float64(mdp["reward"]),                                                                      # nonterminal_reward
            Dict("forward" => Float64(mdp["transition_probabilities"]["forward"]),
                 "left" => Float64(mdp["transition_probabilities"]["left"]),
                 "right" => Float64(mdp["transition_probabilities"]["right"]),
                 "back" => Float64(mdp["transition_probabilities"]["back"])),                            # transition_probabilities
            Float64(mdp["discount_rate"]),                                                               # discount
            Float64(mdp["epsilon"]),                                                                     # epsilon
            Set{State}((x, y) for y in 1:nrows, x in 1:ncols if !((x, y) in blocked)),                   # states
            [West(), North(), South(), East()]                                                           # actions
        )
    end
end

function Base.show(io::IO, mdp::GridWorldMDP)
    println(io, "<GridWorldMDP>")
    println(io, "==                   nrows: ", mdp.nrows)
    println(io, "==                   ncols: ", mdp.ncols)
    println(io, "==                 blocked: ", mdp.blocked)
    println(io, "==         terminal_states: ", mdp.terminal_states)
    println(io, "==        terminal_rewards: ", mdp.terminal_rewards)
    println(io, "==      nonterminal_reward: ", mdp.nonterminal_reward)
    println(io, "==transition_probabilities: ", mdp.transition_probabilities)
    println(io, "==                discount: ", mdp.discount)
    println(io, "==                 epsilon: ", mdp.epsilon)
    println(io, "==                  states: ", mdp.states)
    println(io, "==                 actions: ", mdp.actions)
    print(io, "</GridWorldMDP>")
end

function is_out_of_bounds(mdp::GridWorldMDP, s::State)::Bool
    s[1] < 1 || s[1] > mdp.ncols || s[2] < 1 || s[2] > mdp.nrows || s in mdp.blocked
end

function is_terminal(mdp::GridWorldMDP, s::State)::Bool
    s in mdp.terminal_states
end

function transition(mdp::GridWorldMDP, s::State, a::Action)::State
    if is_terminal(mdp, s) return s end
    successor =
        if a isa North
            (s[1], s[2] + 1)
        elseif a isa South
            (s[1], s[2] - 1)
        elseif a isa East
            (s[1] + 1, s[2])
        else # a isa West
            (s[1] - 1, s[2])
        end
    if is_out_of_bounds(mdp, successor) s else successor end
end

function transition_p(mdp::GridWorldMDP, s::State, a::Action, t::State)::Float64
    if is_terminal(mdp, s)
        if t == s 1.0 else 0.0 end
    else
        rel_dir_to_state = Dict(
            dir => transition(mdp, s, rel_dir_to_action(a)[dir])
            for dir in ["forward", "left", "right", "back"]
        )
        p = 0
        for (reld, sprime) in rel_dir_to_state
            if sprime == t
                p += mdp.transition_probabilities[reld]
            end
        end
        p
    end
end

function reward_f(mdp::GridWorldMDP, s::State, a::Action, t::State)::Float64
    if is_terminal(mdp, s)
        0.0
    elseif is_terminal(mdp, t)
        mdp.terminal_rewards[t]
    else
        mdp.nonterminal_reward
    end
end

initial_values(mdp::GridWorldMDP)::Dict{State, Float64} = Dict(s => 0.0 for s in mdp.states)

all_random_policy(mdp::GridWorldMDP)::Dict{State, Set{Action}} =
    Dict(s => Set(mdp.actions) for s in mdp.states)

uniform_policy_to_general(mdp::GridWorldMDP, policy::Dict{State, Set{Action}})::Dict{Tuple{State, Action}, Float64} =
    Dict(
        (s, a) => if a in policy[s] 1.0 / length(policy[s]) else 0.0 end
        for s in keys(policy)
        for a in mdp.actions
    )

separate() = println()

function pprint_values_grid(mdp::GridWorldMDP, values::Dict{State, Float64})
    cellwidth = max(maximum(length(@sprintf("%.4f", v)) for v in Base.values(values)), 4) + 2
    hsep = "+" * repeat(repeat("-", cellwidth) * "+", mdp.ncols)
    for y in mdp.nrows:-1:1
        println(hsep)
        for x in 1:mdp.ncols
            print("|")
            if (x, y) in mdp.blocked
                lpadding = repeat(" ", (cellwidth - 4) ÷ 2)
                rpadding = repeat(" ", cellwidth - 4 - length(lpadding))
                print(lpadding * "WALL" * rpadding)
            else
                v = get(values, (x, y), 0.0)
                numpad = cellwidth - length(@sprintf("%.4f", v))
                lpadding = repeat(" ", numpad ÷ 2)
                rpadding = repeat(" ", numpad - length(lpadding))
                @printf("%s%.4f%s", lpadding, v, rpadding)
            end
        end
        println("|")
    end
    println(hsep)
end

function pprint_uniform_policy(mdp::GridWorldMDP, uniform_policy::Dict{State, Set{Action}})
    hsep = "+" * repeat(repeat("-", 6) * "+", mdp.ncols)
    for y in mdp.nrows:-1:1
        println(hsep)
        for x in 1:mdp.ncols
            print("|")
            if (x, y) in mdp.blocked
                print(" WALL ")
            elseif is_terminal(mdp, (x, y))
                print(" DONE ")
            else
                best_actions = uniform_policy[(x, y)]
                print(" ")
                for a in mdp.actions
                    if a in best_actions
                        print(a)
                    else
                        print(" ")
                    end
                end
                print(" ")
            end
        end
        println("|")
    end
    println(hsep)
end

function greedy_policy_from(mdp::GridWorldMDP, values::Dict{State, Float64})::Dict{State, Set{Action}}
    policy = Dict{State, Set{Action}}()
    for s in mdp.states
        qvalues =  Dict{Action, Float64}()
        for a in mdp.actions
            qa = 0
            for t in mdp.states
                qa += transition_p(mdp, s, a, t) * (reward_f(mdp, s, a, t) + (mdp.discount * values[t]))
            end
            qvalues[a] = qa
        end
        maxq = maximum(Base.values(qvalues))
        policy[s] = Set(a for (a, q) in qvalues if q == maxq)
    end
    return policy
end

function policy_evaluation_values(mdp::GridWorldMDP, policy::Dict{Tuple{State, Action}, Float64})::Dict{State, Float64}
    values = initial_values(mdp)
    while true
        new_values = Dict{State, Float64}(s => 0.0 for s in mdp.states)
        for s in keys(values)
            for (a, t) in [a => t for a in mdp.actions for t in mdp.states]
                # Backup using Bellman expectation equation
                new_values[s] += policy[(s, a)] * transition_p(mdp, s, a, t) * (
                    reward_f(mdp, s, a, t) + (mdp.discount * values[t])
                )
            end
        end
        if new_values == values break end
        values = new_values
    end
    values
end

function do_policy_iteration(mdp::GridWorldMDP)::Dict{State, Set{Action}}
    policy = all_random_policy(mdp)
    k = 0
    println("Starting policy iteration...")
    while true
        k += 1
        println("==Iteration: ", k, "==")

        # Policy evaluation
        values = policy_evaluation_values(mdp, uniform_policy_to_general(mdp, policy))
        pprint_values_grid(mdp, values)

        # Policy improvement
        greedy = greedy_policy_from(mdp, values)
        pprint_uniform_policy(mdp, greedy)
        
        if greedy == policy break end
        policy = greedy
    end
    return policy
end

function do_value_iteration(mdp::GridWorldMDP)::Dict{State, Float64}
    values = initial_values(mdp)
    k = 0
    println("Starting value iteration...")
    while true
        k += 1
        println("==Iteration: ", k, "==")
        pprint_values_grid(mdp, values)

        new_values = Dict{State, Float64}(s => 0.0 for s in mdp.states)
        for s in keys(values)
            maxq = -Inf
            for a in mdp.actions
                qa = 0.0
                for t in mdp.states
                    qa += transition_p(mdp, s, a, t) * (reward_f(mdp, s, a, t) + (mdp.discount * values[t]))
                end
                maxq = max(maxq, qa)
            end
            # Backup using Bellman optimality equation
            new_values[s] = maxq
        end

        if new_values == values break end
        values = new_values
    end
    values
end

function main()
    if length(ARGS) != 1
        println("Usage: julia mdp.jl <filename>")
        return
    end
    filename = ARGS[1]
    mdp = GridWorldMDP(filename)
    println(mdp)
    separate()
    do_policy_iteration(mdp)
    separate()
    pistar = greedy_policy_from(mdp, do_value_iteration(mdp))
    pprint_uniform_policy(mdp, pistar)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
