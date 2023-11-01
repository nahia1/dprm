module TracyWidom

using LinearAlgebra, SpecialFunctions 

export twcdf, twpdf

function twcdf(s;β=1) 
    β ∈ [1,2] || error("not implemented")
    # 10 point gauss-legendre quadrature nodes and weights
    w0 = [0.06667134430868821, 0.14945134915058056, 0.21908636251598207,
          0.2692667193099965, 0.2955242247147529, 0.2955242247147529, 
          0.2692667193099965, 0.21908636251598207, 0.14945134915058056, 
          0.06667134430868821]
    x0 = [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, 
          -0.4333953941292472, -0.14887433898163122, 0.14887433898163122, 
          0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 
          0.9739065285171717]
    w = sqrt.(w0)
    K = β == 1 ? K1 : K2
    det(I - w*w' .* K.(s,x0',x0))
end

twpdf(s;β=1,h=0.01) = (twcdf(s+h;β)-twcdf(s-h;β))/(2h)

# Trasform integration range from (s,∞) to (-1,1)
ϕ(s,ξ) = s + 10*tan(π*(ξ+1)/4) 
ϕprime(ξ) = 5π/2*sec(π*(ξ+1)/4)^2

function K1(s,ξ,η) 
    x = ϕ(s,ξ) 
    y = ϕ(s,η)
    λ = sqrt(ϕprime(ξ)*ϕprime(η))
    λ * airyai((x+y)/2)/2
end

function K2(s,ξ,η) 
    x = ϕ(s,ξ) 
    y = ϕ(s,η)
    λ = sqrt(ϕprime(ξ)*ϕprime(η))
    if ξ == η
        λ*(airyaiprime(x)^2-x*airyai(x)^2)
    else
        λ*(airyai(x)*airyaiprime(y)-airyaiprime(x)*airyai(y))/(x-y)
    end
end

end
