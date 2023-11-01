module DPRM

using LinearAlgebra, StatsBase, SparseArrays 

export scalingexp,fluct


function transferm(n, d)
    if d == 1
        T = Symmetric(diagm(1 => ones(n-1)))
    elseif d == 2
        T = Symmetric(diagm(1 => ones(n^2-1), n => ones(n^2-n)))
    else
        T = Symmetric(diagm(1 => ones(n^3-1), n => ones(n^3-n),
                            n^2 => ones(n^3-n^2)))
    end
    sparse(T)
end

function dprm(β,T,f,n,l;d=1)
    """
    Calculates the distribution for the 
    endpoint of the DPRM
    Stores the free energy of the polymer in f
    """
    v = zeros(n^d,l)
    v[div(n^d,2)+1] = 1
    for i in 1:l-1
        v[:,i+1] .= T*v[:,i]
        v[:,i+1] .*= exp.(-β*randn(n^d))
        f[i+1] = f[i] + 1/β*log(sum(v[:,i+1]))
        v[:,i+1] ./= sum(v[:,i+1])
    end
    v
end

function fluct(β,n,ls;d=1,nd=100)
    """
    Calculates transverse fluctuations and
    free energy fluctuations over nd realisations
    of randomness
    """
    @assert d ∈ [1,2,3] 
    nlength = length(ls)
    n += mod(n+1,2)
    
    ffluct = zeros(Float64,nlength)
    f = zeros(Float64,nlength,nd)
    x = zeros(Float64,nlength,nd)
    x2 = zeros(Float64,nlength,nd)
    xfluct = zeros(Float64,nlength)
    g = zeros(Float64,nlength)

    T = transferm(n,d)

    coords = reshape(coordinates(n,d),n^d)

    fi = zeros(ls[end])
    Threads.@threads for i in 1:nd
        ρ = dprm(β,T,fi,n,ls[end],d=d)[:,ls]
        for j in 1:nlength
            ρl = ρ[:,j] 
            f[j,i] = fi[ls[j]] 
            x[j,i] = sum(sum.(coords) .* ρl) 
            x2[j,i] = sum(norm.(coords).^2 .* ρl) 
        end
    end

    for j in 1:nlength
        #xfluct[j]  = sqrt(mean(x[j,:].^2)) 
        xfluct[j] = sqrt(mean(x2[j,:]))
        ffluct[j] = std(f[j,:]) 
        g[j] = mean(x[j,:].^2)/mean(x2[j,:]) 
    end

    [xfluct ffluct g]
end

function scalingexp(ls,fluct)
    ζ, c = slm(log.(ls),log.(fluct))
    rmse = rmsd(log.(fluct), c .+ ζ.*log.(ls))
    (ζ,rmse)
end

function slm(x,y)
    """
    Simple linear regression
    """
    slope = cov(x,y)/var(x)
    intercept = mean(y) - slope*mean(x)
    slope, intercept
end

function coordinates(n,d)
    mid = ceil(Int,n/2)
    if d == 1
        [i-mid for i in 1:n]
    elseif d == 2
        [(i-mid,j-mid) for i in 1:n, j in 1:n]
    else
        [(i-mid,j-mid,k-mid) 
                     for i in 1:n, j in 1:n, k in 1:n]
    end
end

end
