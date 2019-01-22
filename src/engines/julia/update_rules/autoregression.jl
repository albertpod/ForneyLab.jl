export ruleVariationalAROutVPPP,
       ruleVariationalARIn1PVPP,
       ruleVariationalARIn2PPVP,
       ruleVariationalARIn3PPPV,
       uvector

order = 10

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

c = uvector(order)
S = shift(order)

function ruleVariationalAROutVPPP(marg_y :: Nothing,
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: ProbabilityDistribution{Univariate})

    m = S*unsafeMean(marg_x)+c*unsafeMean(marg_x)'*unsafeMean(marg_a)
    γ = unsafeMean(marg_γ)*diag()
    Message(GaussianMeanPrecision, m=m, w=γ)
end

function ruleVariationalARIn1PVPP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: Nothing,
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: ProbabilityDistribution{Univariate})

    mA = S+c*unsafeMean(marg_a)'
    m = (unsafeCov(marg_a)+mA'*mA)^-1*mA*unsafeMean(marg_y)
    display(unsafeCov(marg_a))
    display(mA'*mA)
    println(marg_γ)
    γ = (unsafeMean(marg_γ)*unsafeCov(marg_a)+mA'*mA)^-1
    Message(GaussianMeanPrecision, m=m, w=γ)
end

function ruleVariationalARIn2PPVP(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: Nothing,
                                  marg_γ :: ProbabilityDistribution{Univariate})

    D = unsafeCov(marg_x)+unsafeMean(marg_x)*unsafeMean(marg_x)'
    z = unsafeMean(marg_x)*c'unsafeMean(marg_y)-
        unsafeMean(marg_x)*c'*S'*unsafeMean(marg_x)-
        unsafeCov(marg_x)*S*c
    m = D^-1*z
    γ = (unsafeMean(marg_γ)*D)^-1
    Message(GaussianMeanPrecision, m=m, w=γ)
end

function ruleVariationalARIn3PPPV(marg_y :: ProbabilityDistribution{Multivariate},
                                  marg_x :: ProbabilityDistribution{Multivariate},
                                  marg_a :: ProbabilityDistribution{Multivariate},
                                  marg_γ :: Nothing)

    mA = S+c*unsafeMean(marg_a)'
    D(x) = unsafeCov(x) + unsafeMean(x)*unsafeMean(x)'
    AxTy = (mA*unsafeMean(marg_x))'*unsafeMean(marg_y)
    B = tr(D(marg_y)) - tr(AxTy) - tr(AxTy') + tr(D(marg_x)*(S'*mA+unsafeMean(marg_a)*c'*S+D(marg_a)))
    Message(Gamma, a=3/2, b=tr(B)/2)
end
