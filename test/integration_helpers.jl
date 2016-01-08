# This file contains intgration helper functions for constructing and validating context graphs

import Base.==

#############
# Mocks
#############

type MockNode <: Node
    # MockNode is an arbitrary node without update functions
    # The last interface is called :out

    id::Symbol
    interfaces::Array{Interface, 1}
    i::Dict{Symbol, Interface}

    function MockNode(num_interfaces::Int=1; id=ForneyLab.generateNodeId(MockNode))
        self = new(id, Array(Interface, num_interfaces), Dict{Symbol, Interface}())
        addNode!(currentGraph(), self)

        for interface_index = 1:num_interfaces
            self.interfaces[interface_index] = Interface(self)
        end

        self.i[:out] = self.interfaces[end]

        return(self)
    end
end

ForneyLab.isDeterministic(::MockNode) = false # Edge case, same as terminal node

#############
# Backgrounds
#############

function initializePairOfMockNodes()
    # Two unconnected MockNodes
    #
    # [M]--|
    #
    # [M]--|

    g = FactorGraph()
    MockNode(id=:node1)
    MockNode(id=:node2)

    return g
end

function initializePairOfTerminalNodes(d1::ProbabilityDistribution=GaussianDistribution(), d2::ProbabilityDistribution=GaussianDistribution())
    # Two connected TerminalNodes
    #
    # [T]-->[T]

    g = FactorGraph()
    TerminalNode(d1, id=:t1)
    TerminalNode(d2, id=:t2)
    Edge(n(:t1).i[:out], n(:t2).i[:out])
    n(:t1).i[:out].message = Message(d1)
    n(:t2).i[:out].message = Message(d2)

    return g
end

function initializePairOfNodes(; A=[1.0], msg_gain_1=Message(DeltaDistribution(2.0)), msg_gain_2=Message(DeltaDistribution(3.0)), msg_terminal=Message(DeltaDistribution(1.0)))
    # Helper function for initializing an unconnected pair of nodes
    #
    # |--[A]--|
    #
    # |--[C]

    g = FactorGraph()
    GainNode(gain=A, id=:node1)
    n(:node1).interfaces[1].message = msg_gain_1
    n(:node1).interfaces[2].message = msg_gain_2
    TerminalNode(msg_terminal.payload, id=:node2)
    n(:node2).interfaces[1].message = msg_terminal

    return g
end

function initializeChainOfNodes()
    # Chain of three nodes
    #
    #  1     2     3
    # [T]-->[A]-->[B]-->[M]

    g = FactorGraph()
    TerminalNode(DeltaDistribution(3.0), id=:node1)
    GainNode(gain=[2.0], id=:node2)
    GainNode(gain=[2.0], id=:node3)
    Edge(n(:node1).i[:out], n(:node2).i[:in])
    Edge(n(:node2).i[:out], n(:node3).i[:in])
    Edge(n(:node3).i[:out], MockNode().i[:out])

    return g
end

function initializeLoopyGraph(; A=[2.0], B=[0.5], noise_m=1.0, noise_V=0.1)
    # Set up a loopy graph
    #    (driver)
    #   -->[A]---
    #   |       |
    #   |      [+]<-[N]
    #   |       |
    #   ---[B]<--
    #  (inhibitor)

    g = FactorGraph()
    GainNode(gain=A, id=:driver)
    GainNode(gain=B, id=:inhibitor)
    TerminalNode(GaussianDistribution(m=noise_m, V=noise_V), id=:noise)
    AdditionNode(id=:add)
    Edge(n(:add).i[:out], n(:inhibitor).i[:in])
    Edge(n(:inhibitor).i[:out], n(:driver).i[:in])
    Edge(n(:driver).i[:out], n(:add).i[:in1])
    Edge(n(:noise).i[:out], n(:add).i[:in2])

    return g
end

function initializeTreeGraph()
    # Set up some tree graph
    #
    #          (c2)
    #           |
    #           v
    # (c1)---->[+]---->[=]----->
    #                   ^    y
    #                   |
    #                  (c3)
    #

    g = FactorGraph()
    TerminalNode(GaussianDistribution(), id=:c1)
    TerminalNode(GaussianDistribution(), id=:c2)
    TerminalNode(GaussianDistribution(m=-2.0, V=3.0), id=:c3)
    AdditionNode(id=:add)
    EqualityNode(id=:equ)
    # Edges from left to right
    Edge(n(:c1).i[:out], n(:add).i[:in1])
    Edge(n(:c2).i[:out], n(:add).i[:in2])
    Edge(n(:add).i[:out], n(:equ).interfaces[1])
    Edge(n(:c3).i[:out], n(:equ).interfaces[2])
    Edge(n(:c3).i[:out], n(:equ).interfaces[2])

    return g
end

function initializeFactoringGraph()
    # Set up a graph to test factorize function
    #             [T]
    #              |     -----
    #              v     v   |
    # [T]-->[A]-->[N]-->[+] [N]
    #                    |   ^
    #                    -----

    g = FactorGraph()
    TerminalNode(id=:t1)
    GainNode(gain=[1.0], id=:a1)
    GaussianNode(form=:moment, id=:g1)
    TerminalNode(id=:t2)
    AdditionNode(id=:add1)
    GaussianNode(id=:g2, V=0.1)
    Edge(n(:t1).i[:out], n(:a1).i[:in])
    Edge(n(:a1).i[:out], n(:g1).i[:mean])
    Edge(n(:t2).i[:out], n(:g1).i[:variance], InverseGammaDistribution)
    Edge(n(:g1).i[:out], n(:add1).i[:in1])
    Edge(n(:add1).i[:out], n(:g2).i[:mean])
    Edge(n(:g2).i[:out], n(:add1).i[:in2])

    return g
end

function initializeFactoringGraphWithoutLoop()
    # Set up a graph to test factorize function
    #             [T]
    #              |
    #              v
    # [T]-->[A]-->[N]-->[T]
    #
    #

    g = FactorGraph()
    TerminalNode(id=:t1)
    GainNode(gain=[1.0], id=:a1)
    GaussianNode(form=:moment, id=:g1)
    TerminalNode(id=:t2)
    TerminalNode(id=:t3)
    Edge(n(:t1).i[:out], n(:a1).i[:in])
    Edge(n(:a1).i[:out], n(:g1).i[:mean])
    Edge(n(:t2).i[:out], n(:g1).i[:variance], InverseGammaDistribution)
    Edge(n(:g1).i[:out], n(:t3).i[:out])

    return g
end

function initializeGaussianFactoringGraph()
    # [T]<--[A]<--[N]

    g = FactorGraph()
    TerminalNode(id=:t)
    GainNode(gain=[1.0], id=:gain)
    GaussianNode(m=1.0, V=0.5, id=:gauss)
    Edge(n(:gauss).i[:out], n(:gain).i[:in])
    Edge(n(:gain).i[:out], n(:t).i[:out])

    return g
end

function initializeSimpleFactoringGraph()
    # [T]<--[A]<--[T]

    g = FactorGraph()
    TerminalNode(id=:t1)
    GainNode(gain=[1.0], id=:gain)
    TerminalNode(id=:t2)
    Edge(n(:t2).i[:out], n(:gain).i[:in])
    Edge(n(:gain).i[:out], n(:t1).i[:out])

    return g
end

function initializeAdditionNode(values::Array{ProbabilityDistribution})
    # Set up an addition node    #
    # [T]-->[+]<--[T]
    #        |
    #       [T]

    g = FactorGraph()
    AdditionNode(id=:add_node)
    for (id, value) in enumerate(values)
        Edge(TerminalNode(value).i[:out], n(:add_node).interfaces[id])
    end

    return g
end

function initializeEqualityNode(values::Array{ProbabilityDistribution})
    # Set up an equality node
    #
    # [T]-->[=]<--[T] (as many incoming edges as length(msgs))
    #        |
    #       [T]

    g = FactorGraph()
    EqualityNode(length(msgs), id=:eq_node)
    for (id, value) in enumerate(values)
        Edge(TerminalNode(value).i[:out], n(:eq_node).interfaces[id])
    end

    return g
end

function initializeTerminalAndGainAddNode()
    # Initialize some nodes
    #
    #    node
    #    [N]--|
    #       out
    #
    #     c_node
    #    -------
    #    |     |
    # |--|-[+]-|--|
    #    |     |
    #    | ... |

    g = FactorGraph()
    GainAdditionNode([1.0], id=:c_node)
    TerminalNode(id=:node)

    return g
end

function initializeGainAdditionNode(A::Array, values::Array{ProbabilityDistribution})
    # Set up a gain addition node
    #
    #           [T]
    #            | in1
    #            |
    #        ____|____
    #        |   v   |
    #        |  [A]  |
    #        |   |   |
    #    in2 |   v   | out
    #[T]-----|->[+]--|---->[T]
    #        |_______|

    g = FactorGraph()
    GainAdditionNode(A, id=:gac_node)
    for (id, value) in enumerate(values)
        Edge(TerminalNode(value).i[:out], n(:gac_node).interfaces[id])
    end

    return g
end

function initializeTerminalAndGainEqNode()
    # Initialize some nodes
    #
    #    node
    #    [N]--|
    #       out
    #
    #     c_node
    #    -------
    #    |     |
    # |--|-[=]-|--|
    #    |     |
    #    | ... |

    g = FactorGraph()
    GainEqualityNode([1.0], id=:c_node)
    TerminalNode(id=:node)

    return g
end

function initializeGainEqualityNode(A::Array, values::Array{ProbabilityDistribution})
    # Set up a gain equality node and prepare the messages
    # A MockNode is connected for each argument message
    #
    #         _________
    #     in1 |       | in2
    # [T]-----|->[=]<-|-----[T]
    #         |   |   |
    #         |   v   |
    #         |  [A]  |
    #         |___|___|
    #             | out
    #             v
    #            [T]

    g = FactorGraph()
    GainEqualityNode(A, id=:gec_node)
    for (id, value) in enumerate(values)
        Edge(TerminalNode(value).i[:out], n(:gec_node).interfaces[id])
    end

    return g
end

function initializeGaussianNode(; y::ProbabilityDistribution=GaussianDistribution())
    # Initialize a Gaussian node
    #
    #    mean   precision
    #  [T]-->[N]<--[T]
    #         |
    #         v y
    #        [T]

    g = FactorGraph()
    GaussianNode(form=:precision, id=:node)
    Edge(TerminalNode(GaussianDistribution()).i[:out], n(:node).i[:mean], GaussianDistribution, id=:edge1)
    Edge(TerminalNode(GammaDistribution()).i[:out], n(:node).i[:precision], GammaDistribution, id=:edge2)
    Edge(n(:node).i[:out], TerminalNode(y).i[:out], GaussianDistribution, id=:edge3)

    return g
end

function initializeBufferGraph()
    g = FactorGraph()
    TerminalNode(id=:node_t1)
    TerminalNode(id=:node_t2)
    Edge(n(:node_t1), n(:node_t2), id=:e)

    return g
end

function initializeWrapGraph()
    g = FactorGraph()
    TerminalNode(id=:t1)
    TerminalNode(id=:t2)
    TerminalNode(id=:t3)
    TerminalNode(id=:t4)
    AdditionNode(id=:add1)
    AdditionNode(id=:add2)
    Edge(n(:t1), n(:add1).i[:in1])
    Edge(n(:t2), n(:add1).i[:in2])
    Edge(n(:t3), n(:add2).i[:in1])
    Edge(n(:t4), n(:add2).i[:in2])
    Edge(n(:add1).i[:out], n(:add2).i[:out])

    return g
end

function initializeCompositeGraph()
    # Build the internal graph
    g = FactorGraph()
    t_constant = TerminalNode(DeltaDistribution(3.0))
    t_in = TerminalNode(DeltaDistribution(), id=:in)
    t_out = TerminalNode(DeltaDistribution(), id=:out)
    a = AdditionNode(id=:adder)
    Edge(t_in, a.i[:in1])
    Edge(t_constant, a.i[:in2])
    Edge(a.i[:out], t_out)

    return (g, t_in, t_out)
end

function initializeCompositeNode()
    # Build the internal graph
    g = FactorGraph()
    t_constant = TerminalNode(GaussianDistribution(m=3.0, V=1.0))
    t_in = TerminalNode(GaussianDistribution(), id=:in)
    t_out = TerminalNode(GaussianDistribution(), id=:out)
    a = AdditionNode(id=:adder)
    Edge(t_in, a.i[:in1])
    Edge(t_constant, a.i[:in2])
    Edge(a.i[:out], t_out)

    return add_3 = CompositeNode(g, t_in, t_out, id=:add_3)
end

function initializeGaussianNodeChain(y::Array{Float64, 1})
    # Set up a chain of Gaussian nodes for mean-precision estimation
    #
    #     [gam_0]-------->[=]---------->[=]---->    -->[gam_N]
    #                      |             |     etc...
    #     [m_0]-->[=]---------->[=]------------>    -->[m_N]
    #          q(m)| q(gam)|     |       |
    #              -->[N]<--     -->[N]<--
    #                  | q(y)        |
    #                  v             v
    #                [y_1]         [y_2]

    g = FactorGraph()
    # Initial settings
    n_samples = length(y) # Number of observed samples

    # Build graph
    for sec=1:n_samples
        GaussianNode(form=:precision, id=:g*sec)
        EqualityNode(id=:m_eq*sec) # Equality node chain for mean
        EqualityNode(id=:gam_eq*sec) # Equality node chain for precision
        TerminalNode(GaussianDistribution(m=y[sec], V=tiny), id=:y*sec) # Observed y values are stored in terminal node
        Edge(n(:g*sec).i[:out], n(:y*sec).i[:out], GaussianDistribution, id=:q_y*sec)
        Edge(n(:m_eq*sec).i[3], n(:g*sec).i[:mean], GaussianDistribution, id=:q_m*sec)
        Edge(n(:gam_eq*sec).i[3], n(:g*sec).i[:precision], GammaDistribution, id=:q_gam*sec)
        if sec > 1 # Connect sections
            Edge(n(:m_eq*(sec-1)).i[2], n(:m_eq*sec).i[1], GaussianDistribution)
            Edge(n(:gam_eq*(sec-1)).i[2], n(:gam_eq*sec).i[1], GammaDistribution)
        end
    end
    # Attach beginning and end nodes
    TerminalNode(vague(GaussianDistribution), id=:m0) # Prior
    TerminalNode(GammaDistribution(a=1.0-tiny, b=tiny), id=:gam0) # Unifirm prior
    TerminalNode(vague(GaussianDistribution), id=:mN)
    TerminalNode(GammaDistribution(a=1.0-tiny, b=tiny), id=:gamN) # Uniform
    Edge(n(:m0).i[:out], n(:m_eq1).i[1])
    Edge(n(:gam0).i[:out], n(:gam_eq1).i[1])
    Edge(n(:m_eq*n_samples).i[2], n(:mN).i[:out])
    Edge(n(:gam_eq*n_samples).i[2], n(:gamN).i[:out])

    return g
end


#############
# Validations
#############

function ==(x::ScheduleEntry, y::ScheduleEntry)
    if is(x, y) return true end
    ((x.outbound_interface_id == y.outbound_interface_id) && (x.node == y.node) && (x.rule == y.rule)) || (return false)
    (isdefined(x, :post_processing) == isdefined(y, :post_processing)) || (return false)
    if isdefined(x, :post_processing)
        (x.post_processing == y.post_processing) || (return false)
    end
    return true
end

function testInterfaceConnections(node1::GainNode, node2::TerminalNode)
    # Helper function for node comparison

    # Check that nodes are properly connected
    @fact typeof(node1.interfaces[1].message.payload) <: DeltaDistribution --> true
    @fact typeof(node2.interfaces[1].message.payload) <: DeltaDistribution --> true
    @fact mean(node1.interfaces[1].message.payload) --> 2.0
    @fact mean(node2.interfaces[1].message.payload) --> 1.0
    @fact mean(node1.interfaces[1].partner.message.payload) --> 1.0
    @fact mean(node2.interfaces[1].partner.message.payload) --> 2.0
    # Check that pointers are initiatized correctly
    @fact mean(node1.i[:out].message.payload) --> 3.0
    @fact mean(node2.i[:out].message.payload) --> 1.0
    @fact mean(node1.i[:in].partner.message.payload) --> 1.0
    @fact mean(node2.i[:out].partner.message.payload) --> 2.0
end

function validateOutboundMessage(node::Node, outbound_interface_index::Int, inbound_messages::Array, correct_outbound_value::ProbabilityDistribution, update_function::Function=ForneyLab.sumProduct!)
    # Preset an outbound distribution on which the update may operate
    if typeof(correct_outbound_value) <: DeltaDistribution
        outbound_dist = DeltaDistribution()
    elseif typeof(correct_outbound_value) <: MvDeltaDistribution
        outbound_dist = MvDeltaDistribution()
    else
        outbound_dist = vague(typeof(correct_outbound_value))
    end

    # Perform the update and verify the result
    dist = update_function(node, Val{outbound_interface_index}, inbound_messages..., outbound_dist)
    @fact dist --> correct_outbound_value
end
