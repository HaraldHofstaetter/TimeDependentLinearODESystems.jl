using LinearAlgebra
using SparseArrays

export RosenZener, RosenZenerState

mutable struct RosenZener <: TimeDependentSchroedingerMatrix
    V0::Float64
    omega::Float64
    T0::Float64
    d::Int
    H1::SparseMatrixCSC{Float64,Int64}
    H2::SparseMatrixCSC{Complex{Float64},Int64}
    norm_H1::Float64
    norm_H2::Float64
    counter :: Int

    function RosenZener(V0::Real, omega::Real, T0::Real, d::Int)
        k = div(d,2)
        d = 2*k
        S1 =[0.0 1.0
             1.0 0.0]
        S2 = [0.0  -1.0im
              1.0im 0.0]              
        H1 = kron(S1,sparse(1.0I, k, k))
        H2 = kron(S2,spdiagm(-1 => ones(k-1), 1 => ones(k-1)))
        norm_H1 = opnorm(H1, 1)
        norm_H2 = opnorm(H2, 1)

        new(V0, omega, T0, d, H1, H2, norm_H1, norm_H2, 0)
    end
end

Base.show(io::IO, x::RosenZener) = print(io, "RosenZener(V0=$(x.V0), omega=$(x.omega), T0=$(x.T0), dim=$(x.d))")



mutable struct RosenZenerState <: TimeDependentSchroedingerMatrixState
    compute_derivative :: Bool
    c1::Float64
    c2::Float64

    H::RosenZener
end


function (H::RosenZener)(t::Real; compute_derivative::Bool=false)
    if  compute_derivative
        c1 = -H.V0*H.omega*sin(H.omega*t)/cosh(t/H.T0)-H.V0*cos(H.omega*t)*sinh(t/H.T0)/(cosh(t/H.T0)^2*H.T0)
        c2 = -H.V0*H.omega*cos(H.omega*t)/cosh(t/H.T0)+H.V0*sin(H.omega*t)*sinh(t/H.T0)/(cosh(t/H.T0)^2*H.T0)
    else
        c1 = H.V0*cos(H.omega*t)/cosh(t/H.T0)
        c2 = -H.V0*sin(H.omega*t)/cosh(t/H.T0)
    end
    
    RosenZenerState(compute_derivative, c1, c2, H)
end


function (H::RosenZener)(t::Vector{Float64}, c::Vector{Float64};
                  compute_derivative::Bool=false)
    n = length(t)
    @assert n==length(c)&&n>0 "t, c must be vectors of same length>1"
    if  compute_derivative
        c1 = sum([c[j]*(-H.V0*H.omega*sin(H.omega*t[j])/cosh(t[j]/H.T0)-H.V0*cos(H.omega*t[j])*sinh(t[j]/H.T0)/(cosh(t[j]/H.T0)^2*H.T0)) for j=1:n])
        c2 = sum([c[j]*(-H.V0*H.omega*cos(H.omega*t[j])/cosh(t[j]/H.T0)+H.V0*sin(H.omega*t[j])*sinh(t[j]/H.T0)/(cosh(t[j]/H.T0)^2*H.T0)) for j=1:n])
    else
        c1 = sum([c[j]*H.V0*cos(H.omega*t[j])/cosh(t[j]/H.T0) for j=1:n])
        c2 = sum([-c[j]*H.V0*sin(H.omega*t[j])/cosh(t[j]/H.T0) for j=1:n])
    end

    RosenZenerState(compute_derivative, c1, c2, H)
end




function LinearAlgebra.mul!(Y, H::RosenZenerState, B)
    Y[:] = H.c1*(H.H.H1*B) + H.c2*(H.H.H2*B)
    H.H.counter += 1
end


LinearAlgebra.size(H::RosenZener) = (H.d, H.d)
LinearAlgebra.size(H::RosenZener, dim::Int) = dim < 1 ? error("arraysize: dimension out of range") :
                                          (dim < 3 ? H.d : 1)
LinearAlgebra.size(H::RosenZenerState) = size(H.H)
LinearAlgebra.size(H::RosenZenerState, dim::Int) = size(H.H, dim)

LinearAlgebra.eltype(H::RosenZenerState) = Complex{Float64}
LinearAlgebra.issymmetric(H::RosenZenerState) = H.c2==0.0 
LinearAlgebra.ishermitian(H::RosenZenerState) = true 
LinearAlgebra.checksquare(H::RosenZenerState) = H.H.d

function LinearAlgebra.norm(H::RosenZenerState, p::Real=2)
    if p==2
        throw(ArgumentError("2-norm not implemented for RosenZener. Try norm(H, p) where p=1 or Inf."))
    elseif !(p==1 || p==Inf)
        throw(ArgumentError("invalid p-norm p=$p. Valid: 1, Inf"))
    end
    abs(H.c1)*H.H.norm_H1 + abs(H.c2)*H.H.norm_H2
end

full(H::RosenZenerState) = Matrix(H.c1*(H.H.H1) + H.c2*(H.H.H2))



