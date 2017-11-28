module BernoulliTest

using Base.Test
using ForneyLab
import ForneyLab: outboundType, isApplicable, prod!, unsafeMean, unsafeVar, vague, dims
import ForneyLab: SPBernoulliOutVP, VBBernoulliOut

@testset "Bernoulli ProbabilityDistribution and Message construction" begin
    @test ProbabilityDistribution(Univariate, Bernoulli, p=0.2) == ProbabilityDistribution{Univariate, Bernoulli}(Dict(:p=>0.2))
    @test_throws Exception ProbabilityDistribution(Multivariate, Bernoulli)
    @test ProbabilityDistribution(Bernoulli, p=0.2) == ProbabilityDistribution{Univariate, Bernoulli}(Dict(:p=>0.2))
    @test ProbabilityDistribution(Bernoulli) == ProbabilityDistribution{Univariate, Bernoulli}(Dict(:p=>0.5))
    @test Message(Bernoulli) == Message{Bernoulli, Univariate}(ProbabilityDistribution{Univariate, Bernoulli}(Dict(:p=>0.5)))
    @test Message(Univariate, Bernoulli) == Message{Bernoulli, Univariate}(ProbabilityDistribution{Univariate, Bernoulli}(Dict(:p=>0.5)))
    @test_throws Exception Message(Multivariate, Bernoulli)
end

@testset "dims" begin
    @test dims(ProbabilityDistribution(Bernoulli, p=0.5)) == 1
end

@testset "vague" begin
    @test vague(Bernoulli) == ProbabilityDistribution(Bernoulli, p=0.5)
end

@testset "unsafe mean and variance" begin
    @test unsafeMean(ProbabilityDistribution(Bernoulli, p=0.2)) == 0.2
    @test unsafeVar(ProbabilityDistribution(Bernoulli, p=0.5)) == 0.25
end

@testset "prod!" begin
    @test ProbabilityDistribution(Bernoulli, p=0.2) * ProbabilityDistribution(Bernoulli, p=0.8) == ProbabilityDistribution(Bernoulli, p=0.5000000000000001)
    @test_throws Exception ProbabilityDistribution(Bernoulli, p=0.0) * ProbabilityDistribution(Bernoulli, p=1.0)
end

#-------------
# Update rules
#-------------

@testset "SPBernoulliOutVP" begin
    @test SPBernoulliOutVP <: SumProductRule{Bernoulli}
    @test outboundType(SPBernoulliOutVP) == Message{Bernoulli}
    @test isApplicable(SPBernoulliOutVP, [Void, Message{PointMass}]) 

    @test ruleSPBernoulliOutVP(nothing, Message(Univariate, PointMass, m=0.2)) == Message(Univariate, Bernoulli, p=0.2)
end

@testset "VBBernoulliOut" begin
    @test VBBernoulliOut <: VariationalRule{Bernoulli}
    @test outboundType(VBBernoulliOut) == Message{Bernoulli}
    @test isApplicable(VBBernoulliOut, [Void, ProbabilityDistribution])
    @test !isApplicable(VBBernoulliOut, [ProbabilityDistribution, Void])

    @test ruleVBBernoulliOut(nothing, ProbabilityDistribution(Univariate, PointMass, m=0.2)) == Message(Univariate, Bernoulli, p=0.2)
end

@testset "averageEnergy and differentialEntropy" begin
    @test differentialEntropy(ProbabilityDistribution(Univariate, Bernoulli, p=0.25)) == averageEnergy(Bernoulli, ProbabilityDistribution(Univariate, Bernoulli, p=0.25), ProbabilityDistribution(Univariate, PointMass, m=0.25))
end

end # module