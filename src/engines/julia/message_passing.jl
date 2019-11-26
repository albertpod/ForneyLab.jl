function assembleAlgorithm!(rf::RecognitionFactor)
    # Assign message numbers to each interface in the schedule
    interface_to_msg_idx = ForneyLab.interfaceToScheduleEntryIdx(rf.schedule)

    assembleSchedule!(rf.schedule, interface_to_msg_idx)
    assembleInitialization!(rf, interface_to_msg_idx)
    assembleMarginalSchedule!(rf.marginal_schedule, interface_to_msg_idx)

    return rf
end

function assembleSchedule!(schedule::Schedule, interface_to_msg_idx::Dict{Interface, Int})
    # Collect inbounds and assign message index per schedule entry
    for (msg_idx, schedule_entry) in enumerate(schedule)
        schedule_entry.inbounds = collectInbounds(schedule_entry, schedule_entry.message_update_rule, interface_to_msg_idx)
        schedule_entry.schedule_index = msg_idx
    end

    return schedule
end

function assembleInitialization!(rf::RecognitionFactor, interface_to_msg_idx::Dict{Interface, Int})
    # Collect outbound types from schedule
    outbound_types = Dict{Interface, Type}()
    for entry in rf.schedule
        outbound_types[entry.interface] = outboundType(entry.message_update_rule)
    end

    # Find breaker types and dimensions
    rf_update_clamp_flag = false # Flag that tracks whether the update of a clamped variable is required
    rf_initialize_flag = false # Indicate need for initialization block
    for entry in rf.schedule
        partner = ultimatePartner(entry.interface)
        if (entry.message_update_rule <: ExpectationPropagationRule)
            breaker_idx = interface_to_msg_idx[partner]
            breaker_entry = rf.schedule[breaker_idx]
            assembleBreaker!(breaker_entry, family(outbound_types[partner]), ()) # Univariate only
            rf_initialize_flag = true 
        elseif isa(entry.interface.node, Nonlinear) && (entry.interface == entry.interface.node.interfaces[2]) && (entry.interface.node.g_inv == nothing)
            # Set initialization in case of a nonlinear node without given inverse 
            iface = ultimatePartner(entry.interface.node.interfaces[2])
            breaker_idx = interface_to_msg_idx[iface]
            breaker_entry = rf.schedule[breaker_idx]
            assembleBreaker!(breaker_entry, family(outbound_types[iface]), entry.interface.node.dims)
            rf_initialize_flag = true
        elseif !(partner == nothing) && isa(partner.node, Clamp)
            rf_update_clamp_flag = true # Signifies the need for creating a custom `step!` function for optimizing clamped variables
            iface = entry.interface
            breaker_idx = interface_to_msg_idx[iface]
            breaker_entry = rf.schedule[breaker_idx]
            assembleBreaker!(breaker_entry, family(outbound_types[iface]), size(partner.node.value))
            rf_initialize_flag = true
        end
    end

    rf.optimize = rf_update_clamp_flag
    rf.initialize = rf_initialize_flag

    return rf
end

function assembleBreaker!(breaker_entry::ScheduleEntry, family::Type, dimensionality::Tuple)
    breaker_entry.initialize = true
    breaker_entry.dimensionality = dimensionality
    if family == Union{Gamma, Wishart} # Catch special case
        if dimensionality == ()
            breaker_entry.family = ForneyLab.Gamma
        else
            breaker_entry.family = ForneyLab.Wishart
        end
    else
        breaker_entry.family = family
    end

    return breaker_entry
end

function assembleMarginalSchedule!(schedule::MarginalSchedule, interface_to_msg_idx::Dict{Interface, Int})
    for entry in schedule
        if entry.marginal_update_rule == Nothing
            iface = entry.interfaces[1]
            inbounds = [Dict{Symbol, Any}(:schedule_index => interface_to_msg_idx[iface])]
        elseif entry.marginal_update_rule == Product
            iface1 = entry.interfaces[1]
            iface2 = entry.interfaces[2]
            inbounds = [Dict{Symbol, Any}(:schedule_index => interface_to_msg_idx[iface1]),
                        Dict{Symbol, Any}(:schedule_index => interface_to_msg_idx[iface2])]
        else
            inbounds = collectInbounds(entry, interface_to_msg_idx)
        end

        entry.marginal_id = entry.target.id
        entry.inbounds = inbounds
    end

    return schedule
end

"""
Depending on the origin of the Clamp node message, contruct a message inbound
"""
function assembleMessageInbound(node::Clamp{T}) where T<:VariateType
    inbound = Dict{Symbol, Any}(:variate_type => T,
                                :dist_or_msg  => Message)
    if node in keys(ForneyLab.current_graph.placeholders)
        # Message comes from data array
        (buffer, idx) = ForneyLab.current_graph.placeholders[node]
        inbound[:buffer_id] = buffer
        if idx > 0
            inbound[:buffer_index] = idx
        end
    else
        # Insert constant
        inbound[:value] = node.value
    end

    return inbound
end

"""
Depending on the origin of the Clamp node message, contruct a marginal inbound
"""
function assembleMarginalInbound(node::Clamp{T}) where T<:VariateType
    inbound = Dict{Symbol, Any}(:variate_type => T,
                                :dist_or_msg  => ProbabilityDistribution)
    if node in keys(ForneyLab.current_graph.placeholders)
        # Distribution comes from data array
        (buffer, idx) = ForneyLab.current_graph.placeholders[node]
        inbound[:buffer_id] = buffer
        if idx > 0
            inbound[:buffer_index] = idx
        end
    else
        # Insert constant
        inbound[:value] = node.value
    end

    return inbound
end

"""
Construct the inbound code that computes the marginal for `entry`. Allows for
overloading and for a user the define custom node-specific inbounds collection.
Returns a vector with inbounds that correspond with required interfaces.
"""
collectInbounds(entry::MarginalScheduleEntry, interface_to_msg_idx::Dict{Interface, Int}) = collectMarginalNodeInbounds(entry.target.node, entry, interface_to_msg_idx)

function collectMarginalNodeInbounds(::FactorNode, entry::MarginalScheduleEntry, interface_to_msg_idx::Dict{Interface, Int})
    # Collect inbounds
    inbounds = Dict{Symbol, Any}[]
    entry_recognition_factor_id = recognitionFactorId(first(entry.target.edges))
    local_cluster_ids = localRecognitionFactorization(entry.target.node)

    recognition_factor_ids = Symbol[] # Keep track of encountered recognition factor ids
    for node_interface in entry.target.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        partner_node = inbound_interface.node
        node_interface_recognition_factor_id = recognitionFactorId(node_interface.edge)

        if isa(partner_node, Clamp)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleMarginalInbound(partner_node))
        elseif node_interface_recognition_factor_id == entry_recognition_factor_id
            # Collect message from previous result
            inbound_idx = interface_to_msg_idx[inbound_interface]
            push!(inbounds, Dict{Symbol, Any}(:schedule_index => inbound_idx))
        elseif !(node_interface_recognition_factor_id in recognition_factor_ids)
            # Collect marginal from marginal dictionary (if marginal is not already accepted)
            marginal_idx = local_cluster_ids[node_interface_recognition_factor_id]
            push!(inbounds, Dict{Symbol, Any}(:marginal_id => marginal_idx))
        end

        push!(recognition_factor_ids, node_interface_recognition_factor_id)
    end

    return inbounds
end

