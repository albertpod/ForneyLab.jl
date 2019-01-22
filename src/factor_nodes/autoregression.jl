export Autoregression

"""
Description:

    A Gaussian mixture with mean-precision parameterization:

    f(out, a, x, W) = 𝒩(out|Ax, W),

    where A =    a^T
                I	0

Interfaces:

    1. out
    2. a (autoregression coefficients)
    3. x (input vector)
    4. γ (precision)

Construction:

    Autoregression(out, a, x, γ, id=:some_id)
"""
mutable struct Autoregression <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function Autoregression(out, a, x, γ; id=generateId(Autoregression))
        @ensureVariables(out, a, x, γ)
        self = new(id, Array{Interface}(undef, 4), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:a] = self.interfaces[2] = associate!(Interface(self), a)
        self.i[:x] = self.interfaces[3] = associate!(Interface(self), x)
        self.i[:W] = self.interfaces[4] = associate!(Interface(self), γ)
        return self
    end
end

slug(::Type{Autoregression}) = "AR"
