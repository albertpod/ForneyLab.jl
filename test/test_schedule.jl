facts("Schedule and ScheduleEntry tests") do
    context("General properties") do
        FactorGraph()

        # ScheduleEntry
        entry1 = ScheduleEntry(MockNode(id=:mock1), 1, sumProduct!)
        entry2 = ScheduleEntry(MockNode(id=:mock2), 1, sumProduct!)
        @fact is(entry1.node.interfaces[entry1.outbound_interface_id], n(:mock1).i[:out]) --> true
        @fact_throws deepcopy(entry1)

        # copy(::ScheduleEntry)
        entry1_copy = copy(entry1)
        @fact is(entry1, entry1_copy) --> false
        @fact is(entry1.node, entry1_copy.node) --> true
        @fact is(entry1.rule, entry1_copy.rule) --> true
        @fact isdefined(entry1_copy, :post_processing) --> isdefined(entry1, :post_processing)
        if isdefined(entry1_copy, :post_processing)
            @fact is(entry1_copy.post_processing, entry1.post_processing) --> true
        end

        # Schedule
        schedule = [entry1, entry2]
        @fact typeof(schedule) --> Schedule
        schedule_deepcopy = deepcopy(schedule)
        for idx=1:length(schedule)
            @fact is(schedule[idx], schedule_deepcopy[idx]) --> false
            @fact is(schedule[idx].node, schedule_deepcopy[idx].node) --> true
        end
    end

    context("A compiled scheduleEntry can be executed") do
        node = TerminalNode(GaussianDistribution(m=2.0, V=4.0))
        node.i[:out].message = Message(GaussianDistribution()) # Preset message
        entry = ScheduleEntry(node, 1, sumProduct!)
        ForneyLab.buildExecute!(entry, Any[nothing])
        outbound_dist = entry.execute()
        @fact is(node.i[:out].message.payload, outbound_dist) --> true
        @fact node.i[:out].message --> Message(GaussianDistribution(m=2.0, V=4.0))

        node = AdditionNode()
        node.i[:out].message = Message(GaussianDistribution()) # Preset message
        entry = ScheduleEntry(node, 3, sumProduct!)
        ForneyLab.buildExecute!(entry, Any[Message(GaussianDistribution(m=1.0, V=1.0)), Message(GaussianDistribution(m=1.0, V=1.0)), nothing])
        outbound_dist = entry.execute()
        @fact is(node.i[:out].message.payload, outbound_dist) --> true
        @fact node.i[:out].message --> Message(GaussianDistribution(m=2.0, V=2.0))

        node = GaussianNode(form=:precision)
        node.i[:out].message = Message(GaussianDistribution()) # Preset message
        entry = ScheduleEntry(node, 3, vmp!)
        ForneyLab.buildExecute!(entry, Any[GaussianDistribution(m=1.0, V=1.0), GammaDistribution(a=2.0, b=4.0), nothing])
        outbound_dist = entry.execute()
        @fact is(node.i[:out].message.payload, outbound_dist) --> true
        @fact node.i[:out].message --> Message(GaussianDistribution(m=1.0, W=0.5))
    end

    context("A compiled scheduleEntry with post-processing can be executed") do
        node = TerminalNode(GammaDistribution(a=2.0, b=4.0))
        node.i[:out].message = Message(DeltaDistribution()) # Preset message
        entry = ScheduleEntry(node, 1, sumProduct!)
        entry.post_processing = mean
        entry.intermediate_outbound_type = GammaDistribution
        entry.outbound_type = DeltaDistribution{Float64}
        ForneyLab.buildExecute!(entry, Any[nothing])
        outbound_dist = entry.execute()
        @fact is(node.i[:out].message.payload, outbound_dist) --> true
        @fact node.i[:out].message --> Message(DeltaDistribution(0.5))

        node = AdditionNode()
        node.i[:out].message = Message(DeltaDistribution()) # Preset message
        entry = ScheduleEntry(node, 3, sumProduct!)
        entry.post_processing = mean
        entry.intermediate_outbound_type = GaussianDistribution
        entry.outbound_type = DeltaDistribution{Float64}
        ForneyLab.buildExecute!(entry, Any[Message(GaussianDistribution(m=1.0, V=1.0)), Message(GaussianDistribution(m=1.0, V=1.0)), nothing])
        outbound_dist = entry.execute()
        @fact is(node.i[:out].message.payload, outbound_dist) --> true
        @fact node.i[:out].message --> Message(DeltaDistribution(2.0))

        node = GaussianNode(form=:precision)
        node.i[:out].message = Message(DeltaDistribution()) # Preset message
        entry = ScheduleEntry(node, 3, vmp!)
        entry.post_processing = mean
        entry.intermediate_outbound_type = GaussianDistribution
        entry.outbound_type = DeltaDistribution{Float64}
        ForneyLab.buildExecute!(entry, Any[GaussianDistribution(m=1.0, V=1.0), GammaDistribution(a=2.0, b=4.0), nothing])
        outbound_dist = entry.execute()
        @fact is(node.i[:out].message.payload, outbound_dist) --> true
        @fact node.i[:out].message --> Message(DeltaDistribution(1.0))
    end
end
