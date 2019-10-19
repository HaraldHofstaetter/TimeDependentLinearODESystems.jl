module TimeDependentLinearODESystems

using LinearAlgebra

export expmv, expvmv!
export TimeDependentMatrixState, TimeDependentSchroedingerMatrixState
export TimeDependentMatrix, TimeDependentSchroedingerMatrix
export CommutatorFree_Scheme
export CF2, CF4, CF4g6, CF4o, CF6, CF6n, CF6ng8, CF7, CF8, CF8C, CF8AF,  CF10, DoPri45, Magnus4
export get_order, number_of_exponentials
export load_example
export EquidistantTimeStepper, local_orders, local_orders_est
export AdaptiveTimeStepper, EquidistantCorrectedTimeStepper
export global_orders, global_orders_corr


"""
Object representing matrix `B` coming from `A(t)` for concrete t.

**This is not a state vector!**

For derived type `T` One must define:
 - `LinearAlgebra.size(H::T, dim::Int)`: dimensions of matrix
 - `LinearAlgebra.eltype(H::T)`: scalar type (Float64)
 - `LinearAlgebra.issymmetric(H::T)`: true iff matrix is symmetric
 - `LinearAlgebra.ishermitian(H::T)`: true iff matrix is hermitian
 - `LinearAlgebra.checksquare(H::T)`: exception if matrix is not square 
 - `LinearAlgebra.mul!(Y, H::T, B)`: performs `Y = H*B`
"""
abstract type TimeDependentMatrixState end

"""
Object representing matrix `B` coming from `H(t)` for concrete t.

**This is not a state vector!**
"""
abstract type TimeDependentSchroedingerMatrixState <: TimeDependentMatrixState end

"""
Object representing time-dependent matrix `A(t)`.

Given `A::TimeDependentMatrix`, one can do `B=A(t)` to obtain an object
of type `B::TimeDependentMatrixState`, which represents the matrix `A(t)`
evaluated at time `t`.

This is used to solve a ODE of the form  `dv(t)/dt = A(t) v(t)`.
"""
abstract type TimeDependentMatrix end

"""
Object representing time-dependent Hamiltonian `H(t)`.

This is used to solve a ODE of the form  `i dv(t)/dt = H(t) v(t)`.
"""
abstract type TimeDependentSchroedingerMatrix <: TimeDependentMatrix end


import Base.*
function *(A::TimeDependentMatrixState, B) 
    Y = similar(B)
    mul!(Y, A, B)
    Y
end


include("expmv.jl")

load_example(name::String) = include(string(dirname(@__FILE__),"/../examples/",name))


"""
Parameters of integration scheme.

A single time step is then computed from this formula, where `X(t)` is the matrix
`-i H(t)` and `A` and `c` are the parameters of the scheme.

    psi(t+dt) = product(exp(sum(A[j,k] * X(t + c[k]*dt) for k in 1:K)
                        for j in J:-1:1) * psi(t)

**Note:** A is now not the matrix in the ODE, and c are not the coefficients of
the linear combination for evaluating H.

`p` is the order of the scheme.
"""
struct CommutatorFreeScheme
    A::Array{Float64,2}
    c::Array{Float64,1}
    p::Int
end

get_order(scheme::CommutatorFreeScheme) = scheme.p
number_of_exponentials(scheme::CommutatorFreeScheme) = size(scheme.A, 1)
get_lwsp(H, scheme::CommutatorFreeScheme, m::Integer) = min(m, size(H,2))+2 

"""Exponential midpoint rule"""
CF2 = CommutatorFreeScheme( ones(1,1), [1/2], 2 )

CF4 = CommutatorFreeScheme(
    [1/4+sqrt(3)/6 1/4-sqrt(3)/6
     1/4-sqrt(3)/6 1/4+sqrt(3)/6],
    [1/2-sqrt(3)/6, 1/2+sqrt(3)/6],
     4)

CF4g6 = CommutatorFreeScheme(
     [(2*sqrt(15)+5)/36   2/9  (-2*sqrt(15)+5)/36
      (-2*sqrt(15)+5)/36  2/9  (2*sqrt(15)+5)/36],
     [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10],
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

CF6n = CommutatorFreeScheme(
  [ 7.9124225942889763e-01 -8.0400755305553218e-02  1.2765293626634554e-02
   -4.8931475164583259e-01  5.4170980027798808e-02 -1.2069823881924156e-02
   -2.9025638294289255e-02  5.0138457552775674e-01 -2.5145341733509552e-02
    4.8759082890019896e-03 -3.0710355805557892e-02  3.0222764976657693e-01],
  [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10],
  6)

CF6ng8 = CommutatorFreeScheme(
[5.89464201605815655765303416672229380e-01 2.43830015772764437289913379731814221e-01 -1.57259787655929092496086012361705682e-01 4.75723680273279690817865828307895985e-02; -3.65105740082651034389439866942092769e-01 -1.47548276150151315750289379302899394e-01 9.83396340552589170241119484898018497e-02 -3.28992133224145061662174359910648657e-02; -6.86020666155581903549750505993981876e-02 2.89858731629633708796424619465199666e-01 2.91857952484170189124028989070900103e-01 -6.59010219982877682836438241904462939e-02; 1.81710276611204976656434754802610403e-02 -6.00678938209737590225805945051140074e-02 9.31347785477730576614131001900042084e-02 2.25155289862101234054606651961721100e-01],
   [-sqrt(1/140*(2*sqrt(30)+15))+1/2, 
    -sqrt(1/140*(-2*sqrt(30)+15))+1/2,
     sqrt(1/140*(-2*sqrt(30)+15))+1/2, 
     sqrt(1/140*(2*sqrt(30)+15))+1/2],
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




function step!(psi::Union{Array{Float64,1},Array{Complex{Float64},1}}, 
               H::TimeDependentMatrix, 
               t::Real, dt::Real, scheme::CommutatorFreeScheme,
               wsp::Union{Vector{Vector{Float64}},Vector{Vector{Complex{Float64}}}}; 
               expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    tt = t .+ dt*scheme.c
    for j=1:number_of_exponentials(scheme)
        H1 = H(tt, scheme.A[j,:])
        if expmv_tol==0
            psi[:] = exp(dt*full(H1))*psi
        else
            expmv!(psi, dt, H1, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
        end
    end
end  


function step!(psi::Array{Complex{Float64},1}, H::TimeDependentSchroedingerMatrix, 
               t::Real, dt::Real, scheme::CommutatorFreeScheme,
               wsp::Vector{Vector{Complex{Float64}}}; 
               expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    tt = t .+ dt*scheme.c
    for j=1:number_of_exponentials(scheme)
        H1 = H(tt, scheme.A[j,:], matrix_times_minus_i=false)
        if expmv_tol==0
            psi[:] = exp(-1im*dt*full(H1))*psi
        else
            expmv!(psi, -1im*dt, H1, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
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
    mul!(s2, H, u)
    r[:] = s2[:] 
    if p>=1
        #s1=A*u
        mul!(s1, Hd, u)
        r[:] += f1*s1[:] 
    end
    if p>=2
        #s1=B*s1=BAu
        mul!(s1a, H, s1)
        r[:] += f2*s1a[:] 
    end
    if p>=3
        #s1=B*s1=BBAu
        mul!(s1, H, s1a)
        r[:] += f3*s1[:] 
    end
    if p>=4
        #s1=B*s1=BBBAu
        mul!(s1a, H, s1)
        r[:] += f4*s1a[:] 
    end
    if p>=5
        #s1=B*s1=BBBBAu
        mul!(s1, H, s1a)
        r[:] += f5*s1[:] 
    end
    if p>=6
        #s1=B*s1=BBBBBAu
        mul!(s1a, H, s1)
        r[:] += f6*s1a[:] 
    end

    if p>=2
        #s1=A*s2=ABu
        mul!(s1, Hd, s2)
        r[:] -= f2*s1[:] 
    end
    if p>=3
        #s1=B*s1=BABu
        mul!(s1a, H, s1)
        r[:] -= (2*f3)*s1a[:] 
    end
    if p>=4
        #s1=B*s1=BBABu
        mul!(s1, H, s1a)
        r[:] -= (3*f4)*s1[:] 
    end
    if p>=5
        #s1=B*s1=BBBABu
        mul!(s1a, H, s1)
        r[:] -= (4*f5)*s1a[:] 
    end
    if p>=6
        #s1=B*s1=BBBBABu
        mul!(s1, H, s1a)
        r[:] -= (5*f6)*s1[:] 
    end

    if p>=3
        #s2=B*s2=BBu
        mul!(s2a, H, s2)
        #s1=A*s2=ABBu
        mul!(s1, Hd, s2a)
        r[:] += f3*s1
    end
    if p>=4
        #s1=B*s1=BABBu
        mul!(s1a, H, s1)
        r[:] += (3*f4)*s1a
    end
    if p>=5
        #s1=B*s1=BBABBu
        mul!(s1, H, s1a)
        r[:] += (6*f5)*s1
    end
    if p>=6
        #s1=B*s1=BBBABBu
        mul!(s1a, H, s1)
        r[:] += (10*f6)*s1a
    end

    if p>=4
        #s2=B*s2=BBBu
        mul!(s2, H, s2a)
        #s1=A*s2=ABBBu
        ;  mul!(s1, Hd, s2)
        r[:] -= f4*s1
    end
    if p>=5
        #s1=B*s1=BABBBu
        mul!(s1a, H, s1)
        r[:] -= (4*f5)*s1a
    end
    if p>=6
        #s1=B*s1=BBABBBu
        mul!(s1, H, s1a)
        r[:] -= (10*f6)*s1
    end

    if p>=5
        #s2=B*s2=BBBBu
        mul!(s2a, H, s2)
        #s1=A*s2=ABBBBu
        mul!(s1, Hd, s2a)
        r[:] += f5*s1
    end
    if p>=6
        #s1=B*s1=BABBBBu
        mul!(s1a, H, s1)
        r[:] += (5*f6)*s1a
    end

    if p>=6
        #s2=B*s2=BBBBBu
        mul!(s2, H, s2a)
        #s1=A*s2=ABBBBBu
        mul!(s1, Hd, s2)
        r[:] -= f6*s1
    end
end

function CC!(r::Vector{Complex{Float64}},
             H::TimeDependentMatrixState, Hd::TimeDependentMatrixState,
             u::Vector{Complex{Float64}}, sign::Int, dt::Float64, 
             s::Vector{Complex{Float64}}, s1::Vector{Complex{Float64}})
    mul!(s, Hd, u)
    r[:] = 0.5*dt*s[:]
    mul!(s1, H, s)
    r[:] += (sign*dt^2/12)*s1
    mul!(s, H, u)
    r[:] += 0.5*s[:]
    mul!(s1, Hd, s)
    r[:] -= (sign*dt^2/12)*s1
end



function step_estimated_CF2_trapezoidal_rule!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 wsp::Vector{Vector{Complex{Float64}}}; 
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    n = size(H, 2)
    s = wsp[1]

    H1d = H(t+0.5*dt, matrix_times_minus_i=true, compute_derivative=true)
    mul!(psi_est, H1d, psi)
    psi_est[:] *= 0.25*dt

    H1 = H(t+0.5*dt, matrix_times_minus_i=false)
    if expmv_tol==0
        psi[:] = exp(-1im*dt*full(H1))*psi
        psi_est[:] = exp(-1im*dt*full(H1))*psi_est
    else
        expmv!(psi, -1im*dt, H1, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
        expmv!(psi_est, -1im*dt, H1, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
    end

    H1 = H(t+0.5*dt, matrix_times_minus_i=true)
    mul!(s, H1, psi)
    psi_est[:] += s[:]

    H1 = H(t+dt, matrix_times_minus_i=true)
    mul!(s, H1, psi)
    psi_est[:] -= s[:]

    H1d = H(t+0.5*dt, matrix_times_minus_i=true, compute_derivative=true)
    mul!(s, H1d, psi)
    psi_est[:] += 0.25*dt*s[:]

    psi_est[:] *= dt/3
end



function step_estimated_CF2_symm_def!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 wsp::Vector{Vector{Complex{Float64}}}; 
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))

    n = size(H, 2)
    s = wsp[1]

    H1 = H(t, matrix_times_minus_i=true)
    mul!(psi_est, H1, psi)
    psi_est[:] *= -0.5

    H1 = H(t+0.5*dt, matrix_times_minus_i=false)
    if expmv_tol==0
        psi_est[:] = exp(-1im*dt*full(H1))*psi_est
        psi[:] = exp(-1im*dt*full(H1))*psi
    else
        expmv!(psi, -1im*dt, H1, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
        expmv!(psi_est, -1im*dt, H1, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
    end
    
    H1 = H(t+0.5*dt, matrix_times_minus_i=true)
    mul!(s, H1, psi)
    psi_est[:] += s[:]

    H1 = H(t+dt, matrix_times_minus_i=true)
    mul!(s, H1, psi)
    s[:] *= 0.5
    psi_est[:] -= s[:]
    
    psi_est[:] *= dt/3
end


function step_estimated_adjoint_based!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 scheme::CommutatorFreeScheme,
                 wsp::Vector{Vector{Complex{Float64}}}; 
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    tt1 = t .+ dt*scheme.c
    tt2 = t .+ dt*(1.0 .- scheme.c)
    psi_est[:] = psi[:]
    J = number_of_exponentials(scheme)
    for j=1:J
        H1 = H(tt1, scheme.A[j,:], matrix_times_minus_i=false)
        H2 = H(tt2, scheme.A[J+1-j,:], matrix_times_minus_i=false)
        if expmv_tol==0
            psi[:] = exp(-1im*dt*full(H1))*psi
            psi_est[:] = exp(-1im*dt*full(H2))*psi_est
        else
            expmv!(psi, -1im*dt, H1, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
            expmv!(psi_est, -1im*dt, H2, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
        end
    end
    psi_est[:] -= psi[:]
    psi_est[:] *= -0.5
end


function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 scheme::CommutatorFreeScheme,
                 wsp::Vector{Vector{Complex{Float64}}}; 
                 symmetrized_defect::Bool=false, 
                 trapezoidal_rule::Bool=false, 
                 modified_Gamma::Bool=false,
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    if scheme==CF2 && symmetrized_defect
        step_estimated_CF2_symm_def!(psi, psi_est, H, t, dt, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
        return
    elseif scheme==CF2 && trapezoidal_rule
        step_estimated_CF2_trapezoidal_rule!(psi, psi_est, H, t, dt, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
        return
    elseif isodd(get_order(scheme))
        step_estimated_adjoint_based!(psi, psi_est, H, t, dt, scheme, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
        return
    end
    n = size(H, 2)
    s = wsp[1]
    s1 = wsp[2]
    s2 = wsp[3]
    s1a = wsp[4]
    s2a = wsp[5]

    tt = t .+ dt*scheme.c

    if symmetrized_defect
        H1 = H(t, matrix_times_minus_i=true)
        mul!(psi_est, H1, psi)
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
        if expmv_tol==0
            psi[:] = exp(-1im*dt*full(H1e))*psi
            if symmetrized_defect || trapezoidal_rule || j>1
                psi_est[:] = exp(-1im*dt*full(H1e))*psi_est
            end
        else
            expmv!(psi, -1im*dt, H1e, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
            if symmetrized_defect || trapezoidal_rule || j>1
                expmv!(psi_est, -1im*dt, H1e, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
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
    mul!(s, H1, psi)
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
                         wsp::Union{Vector{Vector{Float64}},Vector{Vector{Complex{Float64}}}}; 
                         symmetrized_defect::Bool=false, 
                         trapezoidal_rule::Bool=false, 
                         modified_Gamma::Bool=false,
                         expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1))) where T<:Union{Array{Float64,1},Array{Complex{Float64},1}}
    n = size(H, 2)
    s = wsp[1]
    s1 = wsp[2]
    s2 = wsp[3]
    s1a = wsp[4]
    s2a = wsp[5]

    tt = t+dt*scheme.c

    if symmetrized_defect
        H1 = H(t)
        mul!(psi_est, H1, psi)
        psi_est[:] *= -0.5
        
        H1 = H(tt, scheme.A[1,:])
        if expmv_tol==0
            psi_est[:] = exp(dt*full(H1))*psi_est
        else
            expmv!(psi_est, dt, H1, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
        end
    else
        psi_est[:] .= 0.0
    end
    
    H1 = H(tt, scheme.A[1,:])
    if expmv_tol==0
        psi[:] = exp(dt*full(H1))*psi
    else
        expmv!(psi, dt, H1, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
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
        if expmv_tol==0
            psi_est[:] = exp(dt*full(H1))*psi_est
            psi[:] = exp(dt*full(H1))*psi
        else            
            expmv!(psi, dt, H1, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
            expmv!(psi_est, dt, H1, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
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
    mul!(s, H1, psi)
    #  psi_est = psi_est-s
    psi_est[:] -= s[:]

    # psi_est = psi_est*dt/(p+1)
    psi_est[:] *= dt/(scheme.p+1)

end


abstract type DoPri45 end

get_lwsp(H, scheme::Type{DoPri45}, m::Integer) = 8
get_order(::Type{DoPri45}) = 4

function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentMatrix, 
                 t::Real, dt::Real,
                 scheme::Type{DoPri45},
                 wsp::Vector{Vector{Complex{Float64}}}; 
                 symmetrized_defect::Bool=false,
                 trapezoidal_rule::Bool=false, 
                 modified_Gamma::Bool=false,
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
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
      K = wsp 
      s = K[8]
      for l=1:7
          s[:] = psi
          for j=1:l-1
              if A[l,j]!=0.0
                  s[:] += (dt*A[l,j])*K[j][:]
              end
          end
          H1 = H(t+c[l]*dt)
          mul!(K[l], H1, s)
      end
      psi[:] = s[:]
      s[:] .= 0.0
      for j=1:7
          if e[j]!=0.0
              s[:] += (dt*e[j])*K[j][:]
          end
      end
      H1 = H(t+dt)
      mul!(psi_est, H1, s)
      #psi_est[:] -= psi[:]
      # TODO: K[7] can be reused as K[1] for the next step (FSAL, first same as last)
end


abstract type Magnus4 end

get_lwsp(H, scheme::Type{Magnus4}, m::Integer) = min(m, size(H,2))+4 
get_order(::Type{Magnus4}) = 4
number_of_exponentials(::Type{Magnus4}) = 1

struct Magnus4State <: TimeDependentMatrixState
    H1::TimeDependentMatrixState
    H2::TimeDependentMatrixState
    f_dt::Complex{Float64}
    s::Array{Complex{Float64},1}
end

struct Magnus4DerivativeState <: TimeDependentMatrixState
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


LinearAlgebra.size(H::Magnus4State) = size(H.H1)
LinearAlgebra.size(H::Magnus4State, dim::Int) = size(H.H1, dim) 
LinearAlgebra.eltype(H::Magnus4State) = eltype(H.H1) 
LinearAlgebra.issymmetric(H::Magnus4State) = issymmetric(H.H1) # TODO: check 
LinearAlgebra.ishermitian(H::Magnus4State) = ishermitian(H.H1) # TODO: check 
LinearAlgebra.checksquare(H::Magnus4State) = checksquare(H.H1)

LinearAlgebra.size(H::Magnus4DerivativeState) = size(H.H1)
LinearAlgebra.size(H::Magnus4DerivativeState, dim::Int) = size(H.H1, dim) 
LinearAlgebra.eltype(H::Magnus4DerivativeState) = eltype(H.H1) 
LinearAlgebra.issymmetric(H::Magnus4DerivativeState) = issymmetric(H.H1) # TODO: check 
LinearAlgebra.ishermitian(H::Magnus4DerivativeState) = ishermitian(H.H1) # TODO: check 
LinearAlgebra.checksquare(H::Magnus4DerivativeState) = checksquare(H.H1)



function LinearAlgebra.mul!(Y, H::Magnus4State, B)
    X = H.s 
    mul!(X, H.H1, B)
    Y[:] = 0.5*X[:]
    mul!(X, H.H2, X)
    Y[:] += H.f_dt*X[:]
    mul!(X, H.H2, B)
    Y[:] += 0.5*X[:]
    mul!(X, H.H1, X)
    Y[:] -= H.f_dt*X[:]
end

function full(H::Magnus4State) 
    H1 = full(H.H1)
    H2 = full(H.H2)
    0.5*(H1+H2)-H.f_dt*(H1*H2-H2*H1)
end

function LinearAlgebra.mul!(Y, H::Magnus4DerivativeState, B)
    X = H.s 
    X1 = H.s1

    mul!(X, H.H1d, B)
    Y[:] = (0.5*H.c1)*X[:]
    mul!(X, H.H2, X)
    Y[:] += (H.f*H.c1*H.dt)*X[:]

    mul!(X, H.H2d, B)
    Y[:] += (0.5*H.c2)*X[:] 
    mul!(X, H.H1, X)
    Y[:] -= (H.f*H.c2*H.dt)*X[:]

    mul!(X, H.H1, B)
    mul!(X1, H.H2, X)
    Y[:] += H.f*X1[:]
    mul!(X1, H.H2d, X)
    Y[:] += (H.f*H.c2*H.dt)*X1[:]

    mul!(X, H.H2, B)
    mul!(X1, H.H1, X)
    Y[:] -= H.f*X1[:]
    mul!(X1, H.H1d, X)
    Y[:] -= (H.f*H.c1*H.dt)*X1[:]
end



function step!(psi::Array{Complex{Float64},1}, H::TimeDependentSchroedingerMatrix, 
               t::Real, dt::Real, scheme::Type{Magnus4},
               wsp::Vector{Vector{Complex{Float64}}}; 
               expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    sqrt3 = sqrt(3)
    c1 = 1/2-sqrt3/6
    c2 = 1/2+sqrt3/6
    H1 = H(t + c1*dt, matrix_times_minus_i=false)
    H2 = H(t + c2*dt, matrix_times_minus_i=false)
    f = sqrt3/12
    s = wsp[expmv_m+3]
    HH = Magnus4State(H1, H2, -1im*f*dt, s)
    if expmv_tol==0
        psi[:] = exp(-1im*dt*full(HH))*psi
    else
        expmv!(psi, -1im*dt, HH, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
    end
end  


function step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::TimeDependentSchroedingerMatrix, 
                 t::Real, dt::Real,
                 scheme::Type{Magnus4},
                 wsp::Vector{Vector{Complex{Float64}}}; 
                 symmetrized_defect::Bool=false, 
                 trapezoidal_rule::Bool=false, 
                 modified_Gamma::Bool=false,
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    n = size(H, 2)
    s = wsp[1]
    s1 = wsp[2]
    s2 = wsp[3]
    s1a = wsp[4]
    s2a = wsp[5]
    
    sqrt3 = sqrt(3)
    f = sqrt3/12
    c1 = 1/2-sqrt3/6
    c2 = 1/2+sqrt3/6
    s3 = wsp[expmv_m+3]
    s4 = wsp[expmv_m+4]

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
        mul!(psi_est, H1, psi)
        psi_est[:] *= -0.5
    else
        HHd = Magnus4DerivativeState(H1, H2, H1d, H2d, dt, f, c1, c2, s3, s4)
        psi_est[:] .= 0.0
    end

    if trapezoidal_rule
        CC!(s, HH, HHd, psi, -1, dt, s1, s2)
        psi_est[:] += s[:]

        if expmv_tol==0
            psi[:] = exp(-1im*dt*full(HHe))*psi
            psi_est[:] = exp(-1im*dt*full(HHe))*psi_est
        else
            expmv!(psi, -1im*dt, HHe, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
            expmv!(psi_est, -1im*dt, HHe, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
        end
        
        CC!(s, HH, HHd, psi, +1, dt, s1, s2)
        psi_est[:] += s[:]
    else
        if symmetrized_defect
            if expmv_tol==0
                psi[:] = exp(-1im*dt*full(HHe))*psi
                psi_est[:] = exp(-1im*dt*full(HHe))*psi_est
            else
                expmv!(psi, -1im*dt, HHe, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
                expmv!(psi_est, -1im*dt, HHe, psi_est, tol=expmv_tol, m=expmv_m, wsp=wsp)
            end
        else
            if expmv_tol==0
                psi[:] = exp(-1im*dt*full(HHe))*psi
            else
                expmv!(psi, -1im*dt, HHe, psi, tol=expmv_tol, m=expmv_m, wsp=wsp)
            end
        end
    
        Gamma!(s, HH, HHd, psi, 4, dt, s1, s2, s1a, s2a, modified_Gamma=modified_Gamma)
        psi_est[:] += s[:]
    end

    H1 = H(t + dt, matrix_times_minus_i=true)
    mul!(s, H1, psi)
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
    expmv_tol::Float64
    expmv_m::Int
    wsp  :: Vector{Vector{Complex{Float64}}}  # workspace
    function EquidistantTimeStepper(H::TimeDependentMatrix, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real;
                 scheme=CF4,
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))

        # allocate workspace
        lwsp = get_lwsp(H, scheme, expmv_m)
        wsp = [similar(psi) for k=1:lwsp]
        new(H, psi, t0, tend, dt, scheme, expmv_tol, expmv_m, wsp)
    end
end


function Base.iterate(ets::EquidistantTimeStepper, t=ets.t0)
    if t >= ets.tend
        return nothing
    end
    step!(ets.psi, ets.H, t, ets.dt, ets.scheme, ets.wsp, expmv_tol=ets.expmv_tol, expmv_m=ets.expmv.m)
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
    expmv_tol::Float64
    expmv_m::Int
    psi_est::Array{Complex{Float64},1}
    wsp  :: Vector{Vector{Complex{Float64}}}  # workspace

    function EquidistantCorrectedTimeStepper(H::TimeDependentMatrix, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real; 
                 scheme=CF4,
                 symmetrized_defect::Bool=false,
                 trapezoidal_rule::Bool=false, 
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))

        # allocate workspace
        lwsp = max(5, get_lwsp(H, scheme, expmv_m))
        wsp = [similar(psi) for k=1:lwsp]

        psi_est = zeros(Complex{Float64}, size(H, 2))
        
        new(H, psi, t0, tend, dt, scheme, 
            symmetrized_defect, trapezoidal_rule, expmv_tol, expmv_m, 
            psi_est, wsp)
    end
end


function Base.iterate(ets::EquidistantCorrectedTimeStepper, t=ets.t0)
    if t >= ets.tend
        return nothing
    end
    step_estimated!(ets.psi, ets.psi_est, ets.H, t, ets.dt, ets.scheme, ets.wsp,
                        symmetrized_defect=ets.symmetrized_defect,
                        trapezoidal_rule=ets.trapezoidal_rule,
                        expmv_tol=ets.expmv_tol, expmv_m=ets.expmv_m)
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
                      expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    tab = zeros(Float64, rows, 3)

    # allocate workspace
    lwsp1 = get_lwsp(H, scheme, expmv_m)
    lwsp2 = get_lwsp(H, reference_scheme, expmv_m)
    lwsp = max(lwsp1, lwsp2)
    wsp = [similar(psi) for k=1:lwsp]

    wf_save_initial_value = copy(psi)
    psi_ref = copy(psi)

    dt1 = dt
    err_old = 0.0
    println("             dt         err      p")
    println("-----------------------------------")
    for row=1:rows
        step!(psi, H, t0, dt1, scheme, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
        copyto!(psi_ref, wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, H, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
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
                      expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    tab = zeros(Float64, rows, 5)

    # allocate workspace
    lwsp1 = get_lwsp(H, scheme, expmv_m)
    lwsp2 = get_lwsp(H, reference_scheme, expmv_m)
    lwsp = max(5, lwsp1, lwsp2)
    wsp = [similar(psi) for k=1:lwsp]

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
                        wsp, 
                        symmetrized_defect=symmetrized_defect,
                        trapezoidal_rule=trapezoidal_rule,
                        modified_Gamma=modified_Gamma,
                        expmv_tol=expmv_tol, expmv_m=expmv_m)
        copyto!(psi_ref, wf_save_initial_value)
        dt1_ref = dt1/reference_steps
        for k=1:reference_steps
            step!(psi_ref, H, t0+(k-1)*dt1_ref, dt1_ref, reference_scheme, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
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
                      expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)),
                      corrected_scheme::Bool=false,
                      symmetrized_defect::Bool=false,
                      trapezoidal_rule::Bool=false)
    tab = zeros(Float64, rows, 3)

    # allocate workspace
    lwsp = get_lwsp(H, scheme, expmv_m)
    wsp = [similar(psi) for k=1:lwsp]

    wf_save_initial_value = copy(psi)

    dt1 = dt
    err_old = 0.0
    println("             dt         err           C      p ")
    println("-----------------------------------------------")
    for row=1:rows
        if corrected_scheme
            ets = EquidistantCorrectedTimeStepper(H, psi, t0, tend, dt1, 
                    scheme=scheme, symmetrized_defect=symmetrized_defect,
                    trapezoidal_rule=trapezoidal_rule, expmv_tol=expmv_tol, expmv_m=expmv_m)
        else            
            ets = EquidistantTimeStepper(H, psi, t0, tend, dt1, 
                    scheme=scheme, expmv_tol=expmv_tol, expmv_m=expmv_m)
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
    expmv_tol::Float64
    expmv_m::Int
    psi_est::Array{Complex{Float64},1}
    psi0::Array{Complex{Float64},1}
    wsp  :: Vector{Vector{Complex{Float64}}}  # workspace

    """
    Iterator that steps through ODE solver.
    
        dpsi/dt = A(t) * psi(t)
    
    The local error is defined as:
    
        error = || psi(t+dt) -  psi_est(t+dt) ||_2
    
    Required arguments:
      - `H`:    Matrix for the ODE `A(t) = -i*H(t)`
      - `psi`:  On input, `psi(t0)`, will be updated with `psi(t)`
      - `t0`:   Initial time
      - `tend`: Final time
      - `dt`:   Guess for initial time step
      - `tol`:  Tolerance for local error
    
    Optional arguments:
      - `scheme`:             Integration scheme
      - `symmetrized_defect`: (internal)
      - `trapezoidal_rule`:   (internal)
      - `expmv_tol`:          Tolerance for Lanczos (0: full exp)
    """
    function AdaptiveTimeStepper(H::TimeDependentMatrix, 
                 psi::Array{Complex{Float64},1},
                 t0::Real, tend::Real, dt::Real,  tol::Real; 
                 scheme=CF4,
                 symmetrized_defect::Bool=false,
                 trapezoidal_rule::Bool=false, 
                 expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
        order = get_order(scheme)

        # allocate workspace
        lwsp = max(5, get_lwsp(H, scheme, expmv_m))
        wsp = [similar(psi) for k=1:lwsp]

        psi_est = zeros(Complex{Float64}, size(H, 2))
        psi0 = zeros(Complex{Float64}, size(H, 2))
        
        new(H, psi, t0, tend, dt, tol, order, scheme, 
            symmetrized_defect, trapezoidal_rule, expmv_tol, expmv_m, 
            psi_est, psi0, wsp)
    end
    
end

struct AdaptiveTimeStepperState
   t::Real
   dt::Real
end   

function Base.iterate(ats::AdaptiveTimeStepper, 
                      state::AdaptiveTimeStepperState=AdaptiveTimeStepperState(ats.t0, ats.dt))
    if state.t >= ats.tend
        return nothing
    end

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
        step_estimated!(ats.psi, ats.psi_est, ats.H, state.t, dt, ats.scheme, ats.wsp,
                        symmetrized_defect=ats.symmetrized_defect,
                        trapezoidal_rule=ats.trapezoidal_rule,
                        expmv_tol=ats.expmv_tol, expmv_m=ats.expmv_m)
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
