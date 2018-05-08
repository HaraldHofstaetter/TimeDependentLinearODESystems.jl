module TimeDependentLinearODESystems

using FExpokit

import FExpokit: get_lwsp_liwsp_expv 

get_lwsp_liwsp_expv(H, scheme, m::Integer=30) = get_lwsp_liwsp_expv(size(H, 2), m)

const for_expv = 0
const for_Gamma = 1
const for_Gamma_d = 2
const for_Gamma_d_symmetrized = 3


abstract type TimeDependentMatrix{T} <: AbstractArray{T,2} end

abstract type TimeDependentSchroedingerMatrix{T} <: TimeDependentMatrix{T} end

struct CommutatorFreeScheme
    A::Array{Float64,2}
    c::Array{Float64,1}
    p::Int
end

get_order(scheme::CommutatorFreeScheme) = scheme.p
number_of_exponentials(scheme::CommutatorFreeScheme) = size(scheme.A, 1)

CF2 = CommutatorFreeScheme( ones(1,1), [1/2], 2 )

CF4 = CommutatorFreeScheme(
    [1/4+sqrt(3)/6 1/4-sqrt(3)/6
     1/4-sqrt(3)/6 1/4+sqrt(3)/6],
    [1/2-sqrt(3)/6, 1/2+sqrt(3)/6],
     4)

CF4o = CommutatorFreeScheme(
    [37/240+10/87*sqrt(5/3) -1/30  37/240-10/87*sqrt(5/3)
     -11/360                23/45  -11/360
     37/240-10/87*sqrt(5/3) -1/30  37/240+10/87*sqrt(5/3)],
     [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10],
     4)

CF6 = CommutatorFreeScheme(
  [ 0.2158389969757678 -0.0767179645915514  0.0208789676157837
   -0.0808977963208530 -0.1787472175371576  0.0322633664310473 
    0.1806284600558301  0.4776874043509313 -0.0909342169797981
   -0.0909342169797981  0.4776874043509313  0.1806284600558301
    0.0322633664310473 -0.1787472175371576 -0.0808977963208530 
    0.0208789676157837 -0.0767179645915514  0.2158389969757678],
  [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10],
  6)


function prepare_Omega(A::TimeDependentMatrix, j::Int, which::Int, t::Float64, dt::Float64, scheme::CommutatorFreeScheme)
    set_matrix_times_minus_i!(H, which!=for_expv) 
    if which==for_Gamma_d
       g = 0.0
       f1 = sum(scheme.c.*scheme.A[j,:].*f.(t+dt*scheme.c))
    elseif which==for_Gamma_d_symmetrized
       g = 0.0
       f1 = sum((scheme.c-0.5).*scheme.A[j,:].*f.(t+dt*scheme.c))
    else
       g = sum(scheme.A[j,:])
       f1 = sum(scheme.A[j,:].*f.(t+dt*scheme.c)) 
    end   
    (g, f1)
end

function Omega!(w::Array{Complex{Float64},1}, v::Array{Complex{Float64},1}, H, args::Tuple,
                p::Int, scheme::CommutatorFreeScheme)
    g, f1 = args           
    set_fac!(H, g, f1)
    A_mul_B!(w, H, v) 
end


end module #TimeDependentLinearODESystems
