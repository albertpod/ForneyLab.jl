############################################
# InverseGammaDistribution
############################################
# Description:
#   Encodes an inverse gamma PDF.
#   Pamameters: scalars a (shape) and b (rate).
############################################
export InverseGammaDistribution

type InverseGammaDistribution <: ProbabilityDistribution
    a::Float64 # shape
    b::Float64 # rate
    InverseGammaDistribution(; a=1.0, b=1.0) = new(a, b)
end

uninformative(dist_type::Type{InverseGammaDistribution}) = InverseGammaDistribution(a=-0.999, b=0.001)
function Base.mean(dist::InverseGammaDistribution)
    if dist.a > 1.0
        return dist.b / (dist.a - 1)
    else
        return NaN
    end
end
function Base.var(dist::InverseGammaDistribution)
    if dist.a > 2.0
        return (dist.b^2) / ( ( (dist.a-1)^2 ) * (dist.a-2) )
    else
        return NaN
    end
end

function show(io::IO, dist::InverseGammaDistribution)
    println(io, typeof(dist))
    println(io, "a = $(dist.a) (shape)")
    println(io, "b = $(dist.b) (rate)")
end

function calculateMarginal(forward_dist::InverseGammaDistribution, backward_dist::InverseGammaDistribution)
    return InverseGammaDistribution(a = forward_dist.a+backward_dist.a+1.0, b = forward_dist.b+backward_dist.b)    
end

function calculateMarginal!(edge::Edge, forward_dist::InverseGammaDistribution, backward_dist::InverseGammaDistribution)
    # Calculate the marginal from a forward/backward message pair.
    # We calculate the marginal by using the EqualityNode update rules; same for the functions below
    marg = getOrCreateMarginal(edge, InverseGammaDistribution)
    marg.a = forward_dist.a+backward_dist.a+1.0
    marg.b = forward_dist.b+backward_dist.b
    return marg
end