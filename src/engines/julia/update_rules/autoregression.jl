export ruleVariationalAROutVPPP,
       ruleVariationalARIn1PVPP,
       ruleVariationalARIn2PPVP,
       ruleVariationalARIn3PPPV,
       uvector

import LinearAlgebra.Symmetric

order, c, S = Nothing, Nothing, Nothing

diagAR(dim) = Matrix{Float64}(I, dim, dim)

function shift(dim)
    S = diagAR(dim)
    for i in dim:-1:2
           S[i,:] = S[i-1, :]
    end
    S[1, :] = zeros(dim)
    return S
end

function uvector(dim, pos=1)
    u = zeros(dim)
    u[pos] = 1
    return u
end

function defineOrder(dim)
    global order, c, S
    order = dim
    c = uvector(order)
    S = shift(order)
end

function ruleVariationalAROutVPPP(marg_y :: Nothing,
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: ProbabilityDistribution{Univariate})

    println("@ marg_y")
    ma = unsafeMean(marg_a)
    order == Nothing ? defineOrder(length(ma)) : order
    m = S*unsafeMean(marg_x)+c*unsafeMean(marg_x)'*ma
    W = Symmetric(unsafeMean(marg_w)*diagAR(order))
    display(m)
    display(W)
    Message(Multivariate, GaussianMeanPrecision, m=m, w=W)
end

function ruleVariationalARIn1PVPP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: Nothing,
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: ProbabilityDistribution{Univariate})
    println("@ marg_x")
    ma = unsafeMean(marg_a)
    order == Nothing ? defineOrder(length(ma)) : order
    mA = S+c*ma'
    m = (unsafeCov(marg_a)+mA'*mA)^-1*mA*unsafeMean(marg_y)
    W = Symmetric(unsafeMean(marg_w)*(unsafeCov(marg_a)+mA'*mA))
    display(m)
    display(W)
    Message(Multivariate, GaussianMeanPrecision, m=m, w=W)
end

function ruleVariationalARIn2PPVP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: Nothing,
                                  marg_w :: ProbabilityDistribution{Univariate})
    println("@ marg_a")
    my = unsafeMean(marg_y)
    order == Nothing ? defineOrder(length(mA)) : order
    D = unsafeCov(marg_x)+unsafeMean(marg_x)*unsafeMean(marg_x)'
    z = unsafeMean(marg_x)*c'*my -
        unsafeMean(marg_x)*c'*S'*unsafeMean(marg_x)-
        unsafeCov(marg_x)*S*c
    m = D^-1*z
    W = Symmetric(unsafeMean(marg_w)*D)
    display(m)
    display(W)
    Message(Multivariate, GaussianMeanPrecision, m=m, w=W)
end

function ruleVariationalARIn3PPPV(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_w :: Nothing)

    println("@ marg_w")
    mA = S+c*unsafeMean(marg_a)'
    D(x) = unsafeCov(x) + unsafeMean(x)*unsafeMean(x)'
    AxTy = (mA*unsafeMean(marg_x))'*unsafeMean(marg_y)
    B = tr(D(marg_y)) - tr(AxTy) - tr(AxTy') + tr(D(marg_x)*(S'*mA+unsafeMean(marg_a)*c'*S+D(marg_a)))
    display(tr(B)/2)
    Message(Gamma, a=3/2, b=tr(B)/2)
end
