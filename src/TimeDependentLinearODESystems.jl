module TimeDependentLinearODESystems

using LinearAlgebra
import Base: eltype, size, full

export expmv, expvmv!
export TimeDependentMatrixState, TimeDependentSchroedingerMatrixState
export TimeDependentMatrix, TimeDependentSchroedingerMatrix
export CommutatorFree_Scheme
export CF2, CF4, CF4o, CF6, CF7, CF8, CF8C, CF8AF,  CF10, DoPri45, Magnus4
export get_order, number_of_exponentials
export load_example
export EquidistantTimeStepper, local_orders, local_orders_est
export AdaptiveTimeStepper, EquidistantCorrectedTimeStepper
export global_orders, global_orders_corr
export norm0, Gamma!


abstract type TimeDependentMatrixState end
abstract type TimeDependentSchroedingerMatrixState <: TimeDependentMatrixState end

abstract type TimeDependentMatrix end
abstract type TimeDependentSchroedingerMatrix <: TimeDependentMatrix end

include("expmv.jl")

load_example(name::String) = include(string(dirname(@__FILE__),"/../examples/",name))


#using FExpokit

#import FExpokit: get_lwsp_liwsp_expv 
get_lwsp_liwsp_expv(n::Integer, m::Integer=30) = ( max(10, n*(m+1)+n+(m+2)^2+4*(m+2)^2+6+1), max(7, m+2) )


get_lwsp_liwsp_expv(H, scheme, m::Integer=30) = get_lwsp_liwsp_expv(size(H, 2), m)


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


CF7 = CommutatorFreeScheme(
 [ 2.05862188450411892209e-01    1.69508382914682544509e-01   -1.02088008415028059851e-01    3.04554010755044437431e-02 
  -5.74532495795307023280e-02    2.34286861311879288330e-01    3.32946059487076984706e-01   -7.03703697036401378340e-02
  -8.93040281749440468751e-03    2.71488489365780259156e-02   -2.95144169823456538040e-02   -1.51311830884601959206e-01
   5.52299810755465569835e-01   -3.64425287556240176808e+00    2.53660580449381888484e+00   -6.61436528542997675116e-01
  -5.38241659087501080427e-01    3.60578285850975236760e+00   -2.50685041783117850901e+00    6.51947409253201845106e-01 
   2.03907348473756540850e-02   -6.64014986792173869631e-02    9.49735566789294244299e-02    3.74643341371260411994e-01],  
   [-sqrt(1/140*(2*sqrt(30)+15))+1/2, 
    -sqrt(1/140*(-2*sqrt(30)+15))+1/2,
     sqrt(1/140*(-2*sqrt(30)+15))+1/2, 
     sqrt(1/140*(2*sqrt(30)+15))+1/2],
   7)


CF8 = CommutatorFreeScheme(
 [ 1.84808462624313039047e-01   -2.07206621202004201439e-02    5.02711867953985524846e-03   -1.02882825365674947238e-03
  -2.34494788701189042407e-02    4.21259009948623260268e-01   -4.74878986332597661320e-02    9.04478813619618482626e-03
   4.46203609236170079455e-02   -2.12369356865717369483e-01    5.69989517802253965907e-01    6.02984678266997385471e-03
  -4.93752515735367769884e-02    2.32989476865882554115e-01   -6.22614628245849008467e-01    3.27752279924315371495e-03
   3.27752279924315371495e-03   -6.22614628245849008467e-01    2.32989476865882554115e-01   -4.93752515735367769884e-02
   6.02984678266997385471e-03    5.69989517802253965907e-01   -2.12369356865717369483e-01    4.46203609236170079455e-02
   9.04478813619618482626e-03   -4.74878986332597661320e-02    4.21259009948623260268e-01   -2.34494788701189042407e-02
  -1.02882825365674947238e-03    5.02711867953985524846e-03   -2.07206621202004201439e-02    1.84808462624313039047e-01],
   [-sqrt(1/140*(2*sqrt(30)+15))+1/2, 
    -sqrt(1/140*(-2*sqrt(30)+15))+1/2,
     sqrt(1/140*(-2*sqrt(30)+15))+1/2, 
     sqrt(1/140*(2*sqrt(30)+15))+1/2],
   8)


CF8C = CommutatorFreeScheme( # from CASC paper
 [-1.232611007291861933e+00  1.381999278877963415e-01 -3.352921035850962622e-02  6.861942424401394962e-03 
   1.452637092757343214e+00 -1.632549976033022450e-01  3.986114827352239259e-02 -8.211316003097062961e-03 
  -1.783965547974815151e-02 -8.850494961553933912e-02 -1.299159096777419811e-02  4.448254906109529464e-03 
  -2.982838328015747208e-02  4.530735723950198008e-01 -6.781322579940055086e-03 -1.529505464262590422e-03 
  -1.529505464262590422e-03 -6.781322579940055086e-03  4.530735723950198008e-01 -2.982838328015747208e-02 
   4.448254906109529464e-03 -1.299159096777419811e-02 -8.850494961553933912e-02 -1.783965547974815151e-02 
  -8.211316003097062961e-03  3.986114827352239259e-02 -1.632549976033022450e-01  1.452637092757343214e+00 
   6.861942424401394962e-03 -3.352921035850962622e-02  1.381999278877963415e-01 -1.232611007291861933e+00],
   [-sqrt(1/140*(2*sqrt(30)+15))+1/2, 
    -sqrt(1/140*(-2*sqrt(30)+15))+1/2,
     sqrt(1/140*(-2*sqrt(30)+15))+1/2, 
     sqrt(1/140*(2*sqrt(30)+15))+1/2],
   8)
   

CF8AF = CommutatorFreeScheme(
 [ 1.87122040358115390530e-01   -2.17649338120833602438e-02    5.52892003021124482393e-03   -1.17049553231009501581e-03
   1.20274380119388885065e-03    4.12125752973891079564e-01   -4.12733647828949079769e-02    7.36567552381537106608e-03
   1.35345551498985132129e-01   -5.22856505688516843294e-01    8.04624511929284544063e-01    5.23457489042977401203e-02
  -1.28946403255047767209e-01    4.98263615941272492752e-01   -7.66539274112930211564e-01   -5.10038659643654002820e-02
  -1.13857707205616619581e-02   -1.56116196769613328288e-01   -1.38617787803146844186e-01    1.21952821870042290581e-02
  -2.91430842323998986020e-02    2.52697839525799205662e-01    2.52697839525799205662e-01   -2.91430842323998986020e-02
   1.21952821870042290581e-02   -1.38617787803146844186e-01   -1.56116196769613328288e-01   -1.13857707205616619581e-02
  -5.10038659643654002820e-02   -7.66539274112930211564e-01    4.98263615941272492752e-01   -1.28946403255047767209e-01
   5.23457489042977401203e-02    8.04624511929284544063e-01   -5.22856505688516843294e-01    1.35345551498985132129e-01
   7.36567552381537106608e-03   -4.12733647828949079769e-02    4.12125752973891079564e-01    1.20274380119388885065e-03
  -1.17049553231009501581e-03    5.52892003021124482393e-03   -2.17649338120833602438e-02    1.87122040358115390530e-01],
   [-sqrt(1/140*(2*sqrt(30)+15))+1/2, 
    -sqrt(1/140*(-2*sqrt(30)+15))+1/2,
     sqrt(1/140*(-2*sqrt(30)+15))+1/2, 
     sqrt(1/140*(2*sqrt(30)+15))+1/2],
   8)

   
CF10 = CommutatorFreeScheme(
 [1.257519487460748505e-01 -1.865909914245271482e-02  6.733376258605780510e-03 -2.718352784202390925e-03  7.068483735775850990e-04
 -2.895851111122071992e-03  2.923981849411676845e-01 -4.296189672654135889e-02  1.490123420460316949e-02 -3.808961956414262732e-03
  3.578233942230071908e-02 -2.008488760890393015e-01  6.682006550293361043e-01  8.920336627376998761e-02 -3.910578669555082511e-02
 -1.944671056480696889e-02  1.108002059111070326e-01 -3.720332832453305167e-01 -4.180370067214972631e-02  1.501200997424843587e-02
 -6.498700096451508075e-03 -6.581426937488461349e-03  2.350498736569961365e-01 -8.781176304922676745e-02 -3.978644052337576376e-03
  3.032747599105825582e-02 -3.672352541131558981e-02 -4.341457984155596505e-01  1.431037387565488689e-01 -1.472822613175338694e-02
  1.444710249189639879e-02  9.808293294138223008e-02 -4.072013249310684334e-01  1.240141313609900441e-02  1.992268959805941013e-02
 -3.658914067788396192e-03 -1.244739113166053859e-01  4.885806205957841604e-01 -1.956085512514405479e-03 -2.936517739289611528e-02
 -2.936517739289611528e-02 -1.956085512514405479e-03  4.885806205957841604e-01 -1.244739113166053859e-01 -3.658914067788396192e-03
  1.992268959805941013e-02  1.240141313609900441e-02 -4.072013249310684334e-01  9.808293294138223008e-02  1.444710249189639879e-02
 -1.472822613175338694e-02  1.431037387565488689e-01 -4.341457984155596505e-01 -3.672352541131558981e-02  3.032747599105825582e-02
 -3.978644052337576376e-03 -8.781176304922676745e-02  2.350498736569961365e-01 -6.581426937488461349e-03 -6.498700096451508075e-03
  1.501200997424843587e-02 -4.180370067214972631e-02 -3.720332832453305167e-01  1.108002059111070326e-01 -1.944671056480696889e-02
 -3.910578669555082511e-02  8.920336627376998761e-02  6.682006550293361043e-01 -2.008488760890393015e-01  3.578233942230071908e-02
 -3.808961956414262732e-03  1.490123420460316949e-02 -4.296189672654135889e-02  2.923981849411676845e-01 -2.895851111122071992e-03
  7.068483735775850990e-04 -2.718352784202390925e-03  6.733376258605780510e-03 -1.865909914245271482e-02  1.257519487460748505e-01], 
  ([-sqrt(5+2*sqrt(10/7))/3,
   -sqrt(5-2*sqrt(10/7))/3,
   0.0,
   +sqrt(5-2*sqrt(10/7))/3,
   +sqrt(5+2*sqrt(10/7))/3] .+1)/2,
   10) 



norm0(H::T) where {T<:TimeDependentMatrixState} = norm(H, 1)
#full(H::T) where {T<:TimeDependentMatrixState} = 


function step!(psi::Union{Array{Float64,1},Array{Complex{Float64},1}}, 
               H::TimeDependentMatrix, 
               t::Real, dt::Real, scheme::CommutatorFreeScheme,
               wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1}; #TODO wsp also Array{Float64, 1} !?!?
               use_expm::Bool=false)
    tt = t .+ dt*scheme.c
    for j=1:number_of_exponentials(scheme)
        H1 = H(tt, scheme.A[j,:])
        if use_expm
            psi[:] = exp(dt*full(H1))*psi
        else
            expv!(psi, dt, H1, psi, anorm=norm0(H1), wsp=wsp, iwsp=iwsp)
        end
    end
end  


function step!(psi::Array{Complex{Float64},1}, H::TimeDependentSchroedingerMatrix, 
               t::Real, dt::Real, scheme::CommutatorFreeScheme,
               wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
               use_expm::Bool=false)
    tt = t .+ dt*scheme.c
    for j=1:number_of_exponentials(scheme)
        H1 = H(tt, scheme.A[j,:], matrix_times_minus_i=false)
        if use_expm
            psi[:] = exp(-1im*dt*full(H1))*psi
        else
            expv!(psi, dt, H1, psi, anorm=norm0(H1), 
                  matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
        end
    end
end  

function Gamma!(r::Vector{Complex{Float64}},
                H::TimeDependentMatrixState, Hd::TimeDependentMatrixState,
                u::Vector{Complex{Float64}}, p::Int, dt::Float64, 
                s1::Vector{Complex{Float64}}, s2::Vector{Complex{Float64}},
                s1a::Vector{Complex{Float64}}, s2a::Vector{Complex{Float64}};
                modified_Gamma::Bool=false)
    f1 = dt
    f2 = dt^2/2
    f3 = dt^3/6
    f4 = dt^4/24
    f5 = dt^5/120  
    f6 = dt^6/720
    if modified_Gamma
        if p==2
            p = 3
            f3 = dt^3/4
        elseif p==4
            p = 5
            f5 = dt^5/144
        end
    end
    #s2=B*u
    A_mul_B!(s2, H, u)
    r[:] = s2[:] 
    if p>=1
        #s1=A*u
        A_mul_B!(s1, Hd, u)
        r[:] += f1*s1[:] 
    end
    if p>=2
        #s1=B*s1=BAu
        A_mul_B!(s1a, H, s1)
        r[:] += f2*s1a[:] 
    end
    if p>=3
        #s1=B*s1=BBAu
        A_mul_B!(s1, H, s1a)
        r[:] += f3*s1[:] 
    end
    if p>=4
        #s1=B*s1=BBBAu
        A_mul_B!(s1a, H, s1)
        r[:] += f4*s1a[:] 
    end
    if p>=5
        #s1=B*s1=BBBBAu
        A_mul_B!(s1, H, s1a)
        r[:] += f5*s1[:] 
    end
    if p>=6
        #s1=B*s1=BBBBBAu
        A_mul_B!(s1a, H, s1)
        r[:] += f6*s1a[:] 
    end

    if p>=2
        #s1=A*s2=ABu
        A_mul_B!(s1, Hd, s2)
        r[:] -= f2*s1[:] 
    end
    if p>=3
        #s1=B*s1=BABu
        A_mul_B!(s1a, H, s1)
        r[:] -= (2*f3)*s1a[:] 
    end
    if p>=4
        #s1=B*s1=BBABu
        A_mul_B!(s1, H, s1a)
        r[:] -= (3*f4)*s1[:] 
    end
    if p>=5
        #s1=B*s1=BBBABu
        A_mul_B!(s1a, H, s1)
        r[:] -= (4*f5)*s1a[:] 
    end
    if p>=6
        #s1=B*s1=BBBBABu
        A_mul_B!(s1, H, s1a)
        r[:] -= (5*f6)*s1[:] 
    end

    if p>=3
        #s2=B*s2=BBu
        A_mul_B!(s2a, H, s2)
        #s1=A*s2=ABBu
        A_mul_B!(s1, Hd, s2a)
        r[:] += f3*s1
    end
    if p>=4
        #s1=B*s1=BABBu
        A_mul_B!(s1a, H, s1)
        r[:] += (3*f4)*s1a
    end
    if p>=5
        #s1=B*s1=BBABBu
        A_mul_B!(s1, H, s1a)
        r[:] += (6*f5)*s1
    end
    if p>=6
        #s1=B*s1=BBBABBu
        A_mul_B!(s1a, H, s1)
        r[:] += (10*f6)*s1a
    end

    if p>=4
        #s2=B*s2=BBBu
        A_mul_B!(s2, H, s2a)
        #s1=A*s2=ABBBu
        ;  A_mul_B!(s1, Hd, s2)
        r[:] -= f4*s1
    end
    if p>=5
        #s1=B*s1=BABBBu
        A_mul_B!(s1a, H, s1)
        r[:] -= (4*f5)*s1a
    end
    if p>=6
        #s1=B*s1=BBABBBu
        A_mul_B!(s1, H, s1a)
        r[:] -= (10*f6)*s1
    end

    if p>=5
        #s2=B*s2=BBBBu
        A_mul_B!(s2a, H, s2)
        #s1=A*s2=ABBBBu
        A_mul_B!(s1, Hd, s2a)
        r[:] += f5*s1
    end
    if p>=6
        #s1=B*s1=BABBBBu
        A_mul_B!(s1a, H, s1)
        r[:] += (5*f6)*s1a
    end

    if p>=6
        #s2=B*s2=BBBBBu
        A_mul_B!(s2, H, s2a)
        #s1=A*s2=ABBBBBu
        A_mul_B!(s1, Hd, s2)
        r[:] -= f6*s1
    end
end

function CC!(r::Vector{Complex{Float64}},
             H::TimeDependentMatrixState, Hd::TimeDependentMatrixState,
             u::Vector{Complex{Float64}}, sign::Int, dt::Float64, 
             s::Vector{Complex{Float64}}, s1::Vector{Complex{Float64}})
    A_mul_B!(s, Hd, u)
    r[:] = 0.5*dt*s[:]
    A_mul_B!(s1, H, s)
    r[:] += (sign*dt^2/12)*s1
    A_mul_B!(s, H, u)
    r[:] += 0.5*s[:]
    A_mul_B!(s1, Hd, s)
    r[:] -= (sign*dt^2/12)*s1
end



function step_estimated_CF2_trapezoidal_rule!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
                 use_expm::Bool=false)
    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(wsp, 1), n, own=false)

    H1d = H(t+0.5*dt, matrix_times_minus_i=true, compute_derivative=true)
    A_mul_B!(psi_est, H1d, psi)
    psi_est[:] *= 0.25*dt

    H1 = H(t+0.5*dt, matrix_times_minus_i=false)
    if use_expm
        psi[:] = exp(-1im*dt*full(H1))*psi
        psi_est[:] = exp(-1im*dt*full(H1))*psi_est
    else
        expv!(psi, dt, H1, psi, anorm=norm0(H1), 
              matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
        expv!(psi_est, dt, H1, psi_est, anorm=norm0(H1), 
              matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
    end

    H1 = H(t+0.5*dt, matrix_times_minus_i=true)
    A_mul_B!(s, H1, psi)
    psi_est[:] += s[:]

    H1 = H(t+dt, matrix_times_minus_i=true)
    A_mul_B!(s, H1, psi)
    psi_est[:] -= s[:]

    H1d = H(t+0.5*dt, matrix_times_minus_i=true, compute_derivative=true)
    A_mul_B!(s, H1d, psi)
    psi_est[:] += 0.25*dt*s[:]

    psi_est[:] *= dt/3
end



function step_estimated_CF2_symm_def!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
                 use_expm::Bool=false)

    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(wsp, 1), n, own=false)

    H1 = H(t, matrix_times_minus_i=true)
    A_mul_B!(psi_est, H1, psi)
    psi_est[:] *= -0.5

    H1 = H(t+0.5*dt, matrix_times_minus_i=false)
    if use_expm
        psi_est[:] = exp(-1im*dt*full(H1))*psi_est
        psi[:] = exp(-1im*dt*full(H1))*psi
    else
        expv!(psi_est, dt, H1, psi_est, anorm=norm0(H1), 
              matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
        expv!(psi, dt, H1, psi, anorm=norm0(H1), 
              matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
    end
    
    H1 = H(t+0.5*dt, matrix_times_minus_i=true)
    A_mul_B!(s, H1, psi)
    psi_est[:] += s[:]

    H1 = H(t+dt, matrix_times_minus_i=true)
    A_mul_B!(s, H1, psi)
    s[:] *= 0.5
    psi_est[:] -= s[:]
    
    psi_est[:] *= dt/3
end


function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 scheme::CommutatorFreeScheme,
                 wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
                 symmetrized_defect::Bool=false, 
                 trapezoidal_rule::Bool=false, 
                 modified_Gamma::Bool=false,
                 use_expm::Bool=false)
    if scheme==CF2 && symmetrized_defect
        step_estimated_CF2_symm_def!(psi, psi_est, H, t, dt, wsp, iwsp, use_expm=use_expm)
        return
    elseif scheme==CF2 && trapezoidal_rule
        step_estimated_CF2_trapezoidal_rule!(psi, psi_est, H, t, dt, wsp, iwsp, use_expm=use_expm)
        return
    end
    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(wsp, 1), n, own=false)
    s1 = unsafe_wrap(Array, pointer(wsp, n+1),   n, own=false)
    s2 = unsafe_wrap(Array, pointer(wsp, 2*n+1), n, own=false)
    s1a = unsafe_wrap(Array, pointer(wsp, 3*n+1), n, own=false)
    s2a = unsafe_wrap(Array, pointer(wsp, 4*n+1), n, own=false)

    tt = t .+ dt*scheme.c

    if symmetrized_defect
        H1 = H(t, matrix_times_minus_i=true)
        A_mul_B!(psi_est, H1, psi)
        psi_est[:] *= -0.5
    else
        psi_est[:] .= 0.0
    end

    J = number_of_exponentials(scheme)

    for j=1:J
        H1 = H(tt, scheme.A[j,:], matrix_times_minus_i=true)
        if symmetrized_defect
            H1d = H(tt, (scheme.c .- 0.5).*scheme.A[j,:], compute_derivative=true, matrix_times_minus_i=true)
        else
            H1d = H(tt, scheme.c.*scheme.A[j,:], compute_derivative=true, matrix_times_minus_i=true)
        end
        if trapezoidal_rule 
            CC!(s, H1, H1d, psi, -1, dt, s1, s2)
            psi_est[:] += s[:]
        end

        H1e = H(tt, scheme.A[j,:], matrix_times_minus_i=false)
        if use_expm
            psi[:] = exp(-1im*dt*full(H1e))*psi
            if symmetrized_defect || trapezoidal_rule || j>1
                psi_est[:] = exp(-1im*dt*full(H1e))*psi_est
            end
        else
            expv!(psi, dt, H1e, psi, anorm=norm0(H1e),
                  matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
            if symmetrized_defect || trapezoidal_rule || j>1
                expv!(psi_est, dt, H1e, psi_est, anorm=norm0(H1), 
                      matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
            end
        end
    
        if trapezoidal_rule
            CC!(s, H1, H1d, psi, +1, dt, s1, s2)
        else
            Gamma!(s, H1, H1d, psi, scheme.p, dt, s1, s2, s1a, s2a, modified_Gamma=modified_Gamma)
        end
        psi_est[:] += s[:]
    end
   
    H1 = H(t+dt, matrix_times_minus_i=true)
    A_mul_B!(s, H1, psi)
    if symmetrized_defect
        s[:] *= 0.5
    end
    psi_est[:] -= s[:]

    psi_est[:] *= dt/(scheme.p+1)

end



function step_estimated!(psi::T,
                         psi_est::T,
                         H::TimeDependentMatrix, 
                         t::Real, dt::Real,
                         scheme::CommutatorFreeScheme, 
                         wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1}; #TODO wsp also Array{Float64, 1} !?!?
                         symmetrized_defect::Bool=false, 
                         trapezoidal_rule::Bool=false, 
                         modified_Gamma::Bool=false,
                         use_expm::Bool=false) where T<:Union{Array{Float64,1},Array{Complex{Float64},1}}
    n = size(H, 2)
    s = unsafe_wrap(Array,  pointer(wsp, 1), n, own=false)
    s1 = unsafe_wrap(Array, pointer(wsp, n+1),   n, own=false)
    s2 = unsafe_wrap(Array, pointer(wsp, 2*n+1), n, own=false)
    s1a = unsafe_wrap(Array, pointer(wsp, 3*n+1), n, own=false)
    s2a = unsafe_wrap(Array, pointer(wsp, 4*n+1), n, own=false)

    tt = t+dt*scheme.c

    if symmetrized_defect
        H1 = H(t)
        A_mul_B!(psi_est, H1, psi)
        psi_est[:] *= -0.5
        
        H1 = H(tt, scheme.A[1,:])
        if use_expm
            psi_est[:] = exp(dt*full(H1))*psi_est
        else
            expv!(psi_est, dt, H1, psi_est, anorm=norm0(H1), wsp=wsp, iwsp=iwsp)
        end
    else
        psi_est[:] = 0.0
    end
    
    H1 = H(tt, scheme.A[1,:])
    if use_expm
        psi[:] = exp(dt*full(H1))*psi
    else
        expv!(psi, dt, H1, psi, anorm=norm0(H1), wsp=wsp, iwsp=iwsp)
    end

    if symmetrized_defect
        H1d = H(tt, (scheme.c-0.5).*scheme.A[1,:], compute_derivative=true)
    else
        H1d = H(tt, scheme.c.*scheme.A[1,:], compute_derivative=true)
    end
    Gamma!(s, H1, H1d, psi, scheme.p, dt, s1, s2, s1a, s2a, modified_Gamma=modified_Gamma)
    psi_est[:] += s[:]

    for j=2:number_of_exponentials(scheme)

        H1 = H(tt, scheme.A[j,:])
        if use_expm
            psi_est[:] = exp(dt*full(H1))*psi_est
            psi[:] = exp(dt*full(H1))*psi
        else            
            expv!(psi_est, dt, H1, psi_est, anorm=norm0(H1), wsp=wsp, iwsp=iwsp)
            expv!(psi, dt, H1, psi, anorm=norm0(H1), wsp=wsp, iwsp=iwsp)
        end

        if symmetrized_defect
            H1d = H(tt, (scheme.c-0.5).*scheme.A[j,:], compute_derivative=true)
        else
            H1d = H(tt, scheme.c.*scheme.A[j,:], compute_derivative=true)
        end
        Gamma!(s, H1, H1d, psi, scheme.p, dt, s1, s2, s1a, s2a, modified_Gamma=modified_Gamma)

        psi_est[:] += s[:]

    end
   
    #  s = A(t+dt)*psi
    H1 = H(t+dt)
    A_mul_B!(s, H1, psi)
    #  psi_est = psi_est-s
    psi_est[:] -= s[:]

    # psi_est = psi_est*dt/(p+1)
    psi_est[:] *= dt/(scheme.p+1)

end


abstract type DoPri45 end

get_lwsp_liwsp_expv(H, scheme::Type{DoPri45}, m::Integer=30) = (8*size(H,2), 0)

get_order(::Type{DoPri45}) = 4

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentMatrix, 
                 t::Real, dt::Real,
                 scheme::Type{DoPri45},
                 wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
                 symmetrized_defect::Bool=false,
                 trapezoidal_rule::Bool=false, 
                 modified_Gamma::Bool=false,
                 use_expm::Bool=false)
      c = [0.0 1/5 3/10 4/5 8/9 1.0 1.0]
      A = [0.0         0.0        0.0         0.0      0.0          0.0     0.0
           1/5         0.0        0.0         0.0      0.0          0.0     0.0
           3/40        9/40       0.0         0.0      0.0          0.0     0.0
           44/45      -56/15      32/9        0.0      0.0          0.0     0.0
           19372/6561 -25360/2187 64448/6561 -212/729  0.0          0.0     0.0
           9017/3168  -355/33     46732/5247  49/176  -5103/18656   0.0     0.0
           35/384      0.0        500/1113    125/192 -2187/6784    11/84   0.0]
     # e = [51279/57600 0.0        7571/16695  393/640 -92097/339200 187/2100 1/40]
       e = [71/57600    0.0       -71/16695    71/1920  -17253/339200 22/525 -1/40]    
      n = size(H, 2)
      K = [unsafe_wrap(Array, pointer(wsp, (j-1)*n+1), n, own=false) for j=1:8]
      s = K[8]
      for l=1:7
          s[:] = psi
          for j=1:l-1
              if A[l,j]!=0.0
                  s[:] += (dt*A[l,j])*K[j][:]
              end
          end
          H1 = H(t+c[l]*dt)
          A_mul_B!(K[l], H1, s)
      end
      psi[:] = s[:]
      s[:] = 0.0
      for j=1:7
          if e[j]!=0.0
              s[:] += (dt*e[j])*K[j][:]
          end
      end
      H1 = H(t+dt)
      A_mul_B!(psi_est, H1, s)
      #psi_est[:] -= psi[:]
      # TODO: K[7] can be reused as K[1] for the next step (FSAL, first same as last)
end


abstract type Magnus4 end

function get_lwsp_liwsp_expv(H, scheme::Type{Magnus4}, m::Integer=30) 
    (lw, liw) = get_lwsp_liwsp_expv(size(H, 2), m)
    (lw+size(H, 2), liw)
end

get_order(::Type{Magnus4}) = 4
number_of_exponentials(::Type{Magnus4}) = 1

#The next two structs are mutable for technical reasons: FExpokit calls
#pointer_from_objref on such objects which is only allowed for mutable object.

mutable struct Magnus4State <: TimeDependentMatrixState
    H1::TimeDependentMatrixState
    H2::TimeDependentMatrixState
    f_dt::Complex{Float64}
    s::Array{Complex{Float64},1}
end

mutable struct Magnus4DerivativeState <: TimeDependentMatrixState
    H1::TimeDependentSchroedingerMatrixState
    H2::TimeDependentSchroedingerMatrixState
    H1d::TimeDependentSchroedingerMatrixState
    H2d::TimeDependentSchroedingerMatrixState
    dt::Float64
    f::Complex{Float64}
    c1::Float64
    c2::Float64
    s::Array{Complex{Float64},1}
    s1::Array{Complex{Float64},1}
end


size(H::Magnus4State) = size(H.H1)
size(H::Magnus4State, dim::Int) = size(H.H1, dim) 
eltype(H::Magnus4State) = eltype(H.H1) 
issymmetric(H::Magnus4State) = issymmetric(H.H1) # TODO: check 
ishermitian(H::Magnus4State) = ishermitian(H.H1) # TODO: check 
checksquare(H::Magnus4State) = checksquare(H.H1)

size(H::Magnus4DerivativeState) = size(H.H1)
size(H::Magnus4DerivativeState, dim::Int) = size(H.H1, dim) 
eltype(H::Magnus4DerivativeState) = eltype(H.H1) 
issymmetric(H::Magnus4DerivativeState) = issymmetric(H.H1) # TODO: check 
ishermitian(H::Magnus4DerivativeState) = ishermitian(H.H1) # TODO: check 
checksquare(H::Magnus4DerivativeState) = checksquare(H.H1)



function A_mul_B!(Y, H::Magnus4State, B)
    X = H.s 
    A_mul_B!(X, H.H1, B)
    Y[:] = 0.5*X[:]
    A_mul_B!(X, H.H2, X)
    Y[:] += H.f_dt*X[:]
    A_mul_B!(X, H.H2, B)
    Y[:] += 0.5*X[:]
    A_mul_B!(X, H.H1, X)
    Y[:] -= H.f_dt*X[:]
end

function full(H::Magnus4State) 
    H1 = full(H.H1)
    H2 = full(H.H2)
    0.5*(H1+H2)-H.f_dt*(H1*H2-H2*H1)
end

function A_mul_B!(Y, H::Magnus4DerivativeState, B)
    X = H.s 
    X1 = H.s1

    A_mul_B!(X, H.H1d, B)
    Y[:] = (0.5*H.c1)*X[:]
    A_mul_B!(X, H.H2, X)
    Y[:] += (H.f*H.c1*H.dt)*X[:]

    A_mul_B!(X, H.H2d, B)
    Y[:] += (0.5*H.c2)*X[:] 
    A_mul_B!(X, H.H1, X)
    Y[:] -= (H.f*H.c2*H.dt)*X[:]

    A_mul_B!(X, H.H1, B)
    A_mul_B!(X1, H.H2, X)
    Y[:] += H.f*X1[:]
    A_mul_B!(X1, H.H2d, X)
    Y[:] += (H.f*H.c2*H.dt)*X1[:]

    A_mul_B!(X, H.H2, B)
    A_mul_B!(X1, H.H1, X)
    Y[:] -= H.f*X1[:]
    A_mul_B!(X1, H.H1d, X)
    Y[:] -= (H.f*H.c1*H.dt)*X1[:]
end



function step!(psi::Array{Complex{Float64},1}, H::TimeDependentSchroedingerMatrix, 
               t::Real, dt::Real, scheme::Type{Magnus4},
               wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
               use_expm::Bool=false)
    sqrt3 = sqrt(3)
    c1 = 1/2-sqrt3/6
    c2 = 1/2+sqrt3/6
    H1 = H(t + c1*dt, matrix_times_minus_i=false)
    H2 = H(t + c2*dt, matrix_times_minus_i=false)
    f = sqrt3/12
    s = similar(psi) # TODO: take somthing from wsp
    HH = Magnus4State(H1, H2, -1im*f*dt, s)
    if use_expm
        psi[:] = exp(-1im*dt*full(HH))*psi
    else
        expv!(psi, dt, HH, psi, anorm=norm0(H1), 
             matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
    end
end  


function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 scheme::Type{Magnus4},
                 wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
                 symmetrized_defect::Bool=false, 
                 trapezoidal_rule::Bool=false, 
                 modified_Gamma::Bool=false,
                 use_expm::Bool=false)
    n = size(H, 2)
    s = unsafe_wrap(Array, pointer(wsp, 1), n, own=false)
    s1 = unsafe_wrap(Array, pointer(wsp, n+1),   n, own=false)
    s2 = unsafe_wrap(Array, pointer(wsp, 2*n+1), n, own=false)
    s1a = unsafe_wrap(Array, pointer(wsp, 3*n+1), n, own=false)
    s2a = unsafe_wrap(Array, pointer(wsp, 4*n+1), n, own=false)
    
    sqrt3 = sqrt(3)
    f = sqrt3/12
    c1 = 1/2-sqrt3/6
    c2 = 1/2+sqrt3/6
    s3 = similar(psi) # TODO: take somthing from wsp
    s4 = similar(psi) # TODO: take somthing from wsp

    H1e = H(t + c1*dt, matrix_times_minus_i=false)
    H2e = H(t + c2*dt, matrix_times_minus_i=false)
    HHe = Magnus4State(H1e, H2e, -1im*f*dt, s3)

    H1 = H(t + c1*dt, matrix_times_minus_i=true)
    H2 = H(t + c2*dt, matrix_times_minus_i=true)
    H1d = H(t + c1*dt, matrix_times_minus_i=true, compute_derivative=true)
    H2d = H(t + c2*dt, matrix_times_minus_i=true, compute_derivative=true)
    HH = Magnus4State(H1, H2, f*dt, s3)
    if symmetrized_defect
        HHd = Magnus4DerivativeState(H1, H2, H1d, H2d, dt, f, c1-1/2, c2-1/2, s3, s4)
        H1 = H(t, matrix_times_minus_i=true)
        A_mul_B!(psi_est, H1, psi)
        psi_est[:] *= -0.5
    else
        HHd = Magnus4DerivativeState(H1, H2, H1d, H2d, dt, f, c1, c2, s3, s4)
        psi_est[:] .= 0.0
    end

    if trapezoidal_rule
        CC!(s, HH, HHd, psi, -1, dt, s1, s2)
        psi_est[:] += s[:]

        if use_expm
            psi[:] = exp(-1im*dt*full(HHe))*psi
            psi_est[:] = exp(-1im*dt*full(HHe))*psi_est
        else
            expv!(psi, dt, HHe, psi, anorm=norm0(H1e), 
                 matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
            expv!(psi_est, dt, HHe, psi_est, anorm=norm0(H1e), 
                 matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
        end
        
        CC!(s, HH, HHd, psi, +1, dt, s1, s2)
        psi_est[:] += s[:]
    else
        if symmetrized_defect
           if use_expm
                psi[:] = exp(-1im*dt*full(HHe))*psi
                psi_est[:] = exp(-1im*dt*full(HHe))*psi_est
            else
                expv!(psi, dt, HHe, psi, anorm=norm0(H1e), 
                     matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
                expv!(psi_est, dt, HHe, psi_est, anorm=norm0(H1e), 
                     matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
            end
        else
            if use_expm
                psi[:] = exp(-1im*dt*full(HHe))*psi
            else
                expv!(psi, dt, HHe, psi, anorm=norm0(H1e), 
                     matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
            end
        end
    
        Gamma!(s, HH, HHd, psi, 4, dt, s1, s2, s1a, s2a, modified_Gamma=modified_Gamma)
        psi_est[:] += s[:]
    end

    H1 = H(t + dt, matrix_times_minus_i=true)
    A_mul_B!(s, H1, psi)
    if symmetrized_defect
        s[:] *= 0.5
    end
    psi_est[:] -= s[:]
    psi_est[:] *= (dt/5)
end



struct EquidistantTimeStepper
    H::TimeDependentMatrix
    psi::Array{Complex{Float64},1}
    t0::Float64
    tend::Float64
    dt::Float64
    scheme
    use_expm::Bool
    wsp  :: Array{Complex{Float64},1}  # workspace for expokit
    iwsp :: Array{Int32,1}    # workspace for expokit
    function EquidistantTimeStepper(H::TimeDependentMatrix, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real;
                 scheme=CF4,
                 use_expm::Bool=false)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
        wsp = zeros(Complex{Float64}, lwsp)
        iwsp = zeros(Int32, liwsp) 
        new(H, psi, t0, tend, dt, scheme, use_expm, wsp, iwsp)
    end
end

Base.start(ets::EquidistantTimeStepper) = ets.t0

function Base.done(ets::EquidistantTimeStepper, t) 
    if t >= ets.tend
        return true
    end
    false
end

function Base.next(ets::EquidistantTimeStepper, t)
    step!(ets.psi, ets.H, t, ets.dt, ets.scheme, ets.wsp, ets.iwsp, use_expm=ets.use_expm)
    t1 = t + ets.dt < ets.tend ? t + ets.dt : ets.tend
    t1, t1
end


struct EquidistantCorrectedTimeStepper
    H::TimeDependentMatrix
    psi::Array{Complex{Float64},1}
    t0::Float64
    tend::Float64
    dt::Float64
    scheme
    symmetrized_defect::Bool
    trapezoidal_rule::Bool
    use_expm::Bool
    psi_est::Array{Complex{Float64},1}
    wsp  :: Array{Complex{Float64},1}  # workspace for expokit
    iwsp :: Array{Int32,1}    # workspace for expokit

    function EquidistantCorrectedTimeStepper(H::TimeDependentMatrix, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real; 
                 scheme=CF4,
                 symmetrized_defect::Bool=false,
                 trapezoidal_rule::Bool=false, 
                 use_expm::Bool=false)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
        wsp = zeros(Complex{Float64}, lwsp)
        iwsp = zeros(Int32, liwsp) 

        psi_est = zeros(Complex{Float64}, size(H, 2))
        
        new(H, psi, t0, tend, dt, scheme, 
            symmetrized_defect, trapezoidal_rule, use_expm, 
            psi_est, wsp, iwsp)
    end
end

Base.start(ets::EquidistantCorrectedTimeStepper) = ets.t0

function Base.done(ets::EquidistantCorrectedTimeStepper, t) 
    if t >= ets.tend
        return true
    end
    false
end

function Base.next(ets::EquidistantCorrectedTimeStepper, t)
    step_estimated!(ets.psi, ets.psi_est, ets.H, t, ets.dt, ets.scheme, ets.wsp, ets.iwsp,
                        symmetrized_defect=ets.symmetrized_defect,
                        trapezoidal_rule=ets.trapezoidal_rule,
                        use_expm=ets.use_expm)
    ets.psi[:] -= ets.psi_est # corrected scheme                        
    t1 = t + ets.dt < ets.tend ? t + ets.dt : ets.tend
    t1, t1
end




using Printf

function local_orders(H::TimeDependentMatrix,
                      psi::Array{Complex{Float64},1}, t0::Real, dt::Real; 
                      scheme=CF2, reference_scheme=scheme, 
                      reference_steps=10,
                      rows=8,
                      use_expm::Bool=false)
    tab = zeros(Float64, rows, 3)

    # allocate workspace
    lwsp1, liwsp1 = get_lwsp_liwsp_expv(H, scheme)  
    lwsp2, liwsp2 = get_lwsp_liwsp_expv(H, reference_scheme)  
    lwsp = max(lwsp1, lwsp2)
    liwsp = max(liwsp1, liwsp2)
    wsp = zeros(Complex{Float64}, lwsp)
    iwsp = zeros(Int32, liwsp) 

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)

    dt1 = dt
    err_old = 0.0
    println("             dt         err      p")
    println("-----------------------------------")
    for row=1:rows
        step!(psi, H, t0, dt1, scheme, wsp, iwsp, use_expm=use_expm)
        copyto!(psi_ref, wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, H, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme, wsp, iwsp, use_expm=use_expm)
        end    
        err = norm(psi-psi_ref)
        if (row==1) 
            @printf("%3i%12.3e%12.3e\n", row, Float64(dt1), Float64(err))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = 0 
        else
            p = log(err_old/err)/log(2.0);
            @printf("%3i%12.3e%12.3e%7.2f\n", row, Float64(dt1), Float64(err), Float64(p))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = p 
        end
        err_old = err
        dt1 = 0.5*dt1
        copyto!(psi, wf_save_initial_value)
    end

    tab
end

function local_orders_est(H::TimeDependentMatrix,
                      psi::Array{Complex{Float64},1}, t0::Real, dt::Real; 
                      scheme=CF2_defectbased, reference_scheme=CF4, 
                      reference_steps=10,
                      symmetrized_defect::Bool=false,
                      trapezoidal_rule::Bool=false,
                      modified_Gamma::Bool=false,
                      rows=8,
                      use_expm::Bool=false)
    tab = zeros(Float64, rows, 5)

    # allocate workspace
    lwsp1, liwsp1 = get_lwsp_liwsp_expv(H, scheme)  
    lwsp2, liwsp2 = get_lwsp_liwsp_expv(H, reference_scheme)  
    lwsp = max(lwsp1, lwsp2)
    liwsp = max(liwsp1, liwsp2)
    wsp = zeros(Complex{Float64}, lwsp)
    iwsp = zeros(Int32, liwsp) 

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)
    psi_est = copy(psi)

    dt1 = dt
    err_old = 0.0
    err_est_old = 0.0
    println("             dt         err      p       err_est      p")
    println("--------------------------------------------------------")
    for row=1:rows
        step_estimated!(psi, psi_est, H, t0, dt1, scheme,
                        wsp, iwsp,
                        symmetrized_defect=symmetrized_defect,
                        trapezoidal_rule=trapezoidal_rule,
                        modified_Gamma=modified_Gamma,
                        use_expm=use_expm)
        copyto!(psi_ref, wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, H, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme, wsp, iwsp, use_expm=use_expm)
        end    
        err = norm(psi-psi_ref)
        err_est = norm(psi-psi_ref-psi_est)
        if (row==1) 
            @printf("%3i%12.3e%12.3e  %19.3e\n", row, Float64(dt1), Float64(err), Float64(err_est))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = 0 
            tab[row,4] = err_est
            tab[row,5] = 0 
        else
            p = log(err_old/err)/log(2.0);
            p_est = log(err_est_old/err_est)/log(2.0);
            @printf("%3i%12.3e%12.3e%7.2f  %12.3e%7.2f\n", 
                    row, Float64(dt1), Float64(err), Float64(p), 
                                       Float64(err_est), Float64(p_est))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = p 
            tab[row,4] = err_est
            tab[row,5] = p_est 
        end
        err_old = err
        err_est_old = err_est
        dt1 = 0.5*dt1
        copyto!(psi, wf_save_initial_value)
    end

    tab
end


function global_orders(H::TimeDependentMatrix,
                      psi::Array{Complex{Float64},1}, 
                      psi_ref::Array{Complex{Float64},1}, 
                      t0::Real, tend::Real, dt::Real; 
                      scheme=CF2, rows=8,
                      use_expm::Bool=false,
                      corrected_scheme::Bool=false,
                      symmetrized_defect::Bool=false,
                      trapezoidal_rule::Bool=false)
    tab = zeros(Float64, rows, 3)

    # allocate workspace
    lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
    wsp = zeros(Complex{Float64}, lwsp)
    iwsp = zeros(Int32, liwsp) 

    wf_save_initial_value = copy(psi)

    dt1 = dt
    err_old = 0.0
    println("             dt         err           C      p ")
    println("-----------------------------------------------")
    for row=1:rows
        if corrected_scheme
            ets = EquidistantCorrectedTimeStepper(H, psi, t0, tend, dt1, 
                    scheme=scheme, symmetrized_defect=symmetrized_defect,
                    trapezoidal_rule=trapezoidal_rule, use_expm=use_expm)
        else            
            ets = EquidistantTimeStepper(H, psi, t0, tend, dt1, 
                    scheme=scheme, use_expm=use_expm)
        end
        for t in ets end
        err = norm(psi-psi_ref)
        if (row==1) 
            @Printf.printf("%3i%12.3e%12.3e\n", row, Float64(dt1), Float64(err))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = 0 
        else
            p = log(err_old/err)/log(2.0)
            C = err/dt1^p
            @Printf.printf("%3i%12.3e%12.3e%12.3e%7.2f\n", row, Float64(dt1), Float64(err),
                                                       Float64(C), Float64(p))
            tab[row,1] = dt1
            tab[row,2] = err
            tab[row,3] = p 
        end
        err_old = err
        dt1 = 0.5*dt1
        copyto!(psi, wf_save_initial_value)
    end
end


struct AdaptiveTimeStepper
    H::TimeDependentMatrix
    psi::Array{Complex{Float64},1}
    t0::Float64
    tend::Float64
    dt::Float64
    tol::Float64
    order::Int
    scheme
    symmetrized_defect::Bool
    trapezoidal_rule::Bool
    use_expm::Bool
    psi_est::Array{Complex{Float64},1}
    psi0::Array{Complex{Float64},1}
    wsp  :: Array{Complex{Float64},1}  # workspace for expokit
    iwsp :: Array{Int32,1}    # workspace for expokit

    function AdaptiveTimeStepper(H::TimeDependentMatrix, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real,  tol::Real; 
                 scheme=CF4,
                 symmetrized_defect::Bool=false,
                 trapezoidal_rule::Bool=false, 
                 use_expm::Bool=false)
        order = get_order(scheme)

        # allocate workspace
        lwsp, liwsp = get_lwsp_liwsp_expv(H, scheme)  
        wsp = zeros(Complex{Float64}, lwsp)
        iwsp = zeros(Int32, liwsp) 

        psi_est = zeros(Complex{Float64}, size(H, 2))
        psi0 = zeros(Complex{Float64}, size(H, 2))
        
        new(H, psi, t0, tend, dt, tol, order, scheme, 
            symmetrized_defect, trapezoidal_rule, use_expm, 
            psi_est, psi0, wsp, iwsp)
    end
    
end

struct AdaptiveTimeStepperState
   t::Real
   dt::Real
end   

Base.start(ats::AdaptiveTimeStepper) = AdaptiveTimeStepperState(ats.t0, ats.dt)

function Base.done(ats::AdaptiveTimeStepper, state::AdaptiveTimeStepperState)
    if state.t >= ats.tend
        return true
    end
    false
end  

function Base.next(ats::AdaptiveTimeStepper, state::AdaptiveTimeStepperState)
    facmin = 0.25
    facmax = 4.0
    fac = 0.9

    dt = state.dt
    dt0 = dt
    ats.psi0[:] = ats.psi[:]
    err = 2.0
    while err>=1.0
        dt = min(dt, ats.tend-state.t)
        dt0 = dt
        step_estimated!(ats.psi, ats.psi_est, ats.H, state.t, dt, ats.scheme, ats.wsp, ats.iwsp,
                        symmetrized_defect=ats.symmetrized_defect,
                        trapezoidal_rule=ats.trapezoidal_rule,
                        use_expm=ats.use_expm)
        err = norm(ats.psi_est)/ats.tol
        dt = dt*min(facmax, max(facmin, fac*(1.0/err)^(1.0/(ats.order+1))))
        if err>=1.0
           ats.psi[:] = ats.psi0
           @printf("t=%17.9e  err=%17.8e  dt=%17.8e  rejected...\n", Float64(state.t), Float64(err), Float64(dt))
        end   
    end
    state.t + dt0, AdaptiveTimeStepperState(state.t+dt0, dt)
end






end #TimeDependentLinearODESystems
