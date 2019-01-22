@naiveVariationalRule(:node_type     => Autoregression,
                      :outbound_type => Message{GaussianMeanVariance},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalAROutVPPPP)


@naiveVariationalRule(:node_type     => Autoregression,
                      :outbound_type => Message{GaussianMeanVariance},
                      :inbound_types => (ProbabilityDistribution, Nothing, ProbabilityDistribution, ProbabilityDistribution),
                      :name          => VariationalARIn1PVPP)

@naiveVariationalRule(:node_type     => Autoregression,
                      :outbound_type => Message{GaussianMeanVariance},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, Nothing, ProbabilityDistribution),
                      :name          => VariationalARIn2PPVP)

@naiveVariationalRule(:node_type     => Autoregression,
                      :outbound_type => Message{GaussianMeanVariance},
                      :inbound_types => (ProbabilityDistribution, ProbabilityDistribution, ProbabilityDistribution, Nothing),
                      :name          => VariationalARIn3PPPV)
