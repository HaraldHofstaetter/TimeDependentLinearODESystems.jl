load_example("hubbard.jl")


function get_integral_f(t::Real, dt::Real, f::Function; with_factor::Bool=false)
    #order 4
    xw = [(1/2-sqrt(1/12), 1/2),
          (1/2+sqrt(1/12), 1/2)]
    #order 6
    #xw = [(1/2-sqrt(3/20), 5/18),
    #      (1/2           , 8/18),
    #      (1/2+sqrt(3/20), 5/18)] 
    if with_factor
        return dt^2*sum([w*f(t+dt*x)*(x-1/2) for (x,w) in xw])

    else
        return dt*sum([w*f(t+dt*x) for (x,w) in xw])
    end    
end


function get_integral_r(t::Real, dt::Real, f::Function)
    # order 4:
    xyw=[(0.445948490915965, 0.445948490915965, 0.223381589678011),
         (0.445948490915965, 0.108103018168070, 0.223381589678011),
         (0.108103018168070, 0.445948490915965, 0.223381589678011),
         (0.091576213509771, 0.091576213509771, 0.109951743655322),
         (0.091576213509771, 0.816847572980459, 0.109951743655322),
         (0.816847572980459, 0.091576213509771, 0.109951743655322)]
    
    # order 6:
    #xyw=[(0.333333333333333, 0.333333333333333, 0.225000000000000),
    #     (0.470142064105115, 0.470142064105115, 0.132394152788506),
    #     (0.470142064105115, 0.059715871789770, 0.132394152788506),
    #     (0.059715871789770, 0.470142064105115, 0.132394152788506),
    #     (0.101286507323456, 0.101286507323456, 0.125939180544827),
    #     (0.101286507323456, 0.797426985353087, 0.125939180544827),
    #     (0.797426985353087, 0.101286507323456, 0.125939180544827)]; 
    h = 0.0
    for (x,y,w) in xyw
        tx = t+dt*x
        ty = t+dt*(1-y)
        fx = f(tx)
        fy = f(ty)
        cx,sx = real(fx), imag(fx)
        cy,sy = real(fy), imag(fy)
        h += w*(cy*sx-cx*sy)
    end
    0.25*dt^2*h
end

function get_integrals_csr(t::Real, dt::Real, f::Function)
    # order 4:
    xyw=[(0.445948490915965, 0.445948490915965, 0.223381589678011),
         (0.445948490915965, 0.108103018168070, 0.223381589678011),
         (0.108103018168070, 0.445948490915965, 0.223381589678011),
         (0.091576213509771, 0.091576213509771, 0.109951743655322),
         (0.091576213509771, 0.816847572980459, 0.109951743655322),
         (0.816847572980459, 0.091576213509771, 0.109951743655322)]
    
    # order 6:
    #xyw=[(0.333333333333333, 0.333333333333333, 0.225000000000000),
    #     (0.470142064105115, 0.470142064105115, 0.132394152788506),
    #     (0.470142064105115, 0.059715871789770, 0.132394152788506),
    #     (0.059715871789770, 0.470142064105115, 0.132394152788506),
    #     (0.101286507323456, 0.101286507323456, 0.125939180544827),
    #     (0.101286507323456, 0.797426985353087, 0.125939180544827),
    #     (0.797426985353087, 0.101286507323456, 0.125939180544827)]; 
    c = 0.0
    s = 0.0
    r = 0.0
    for (x,y,w) in xyw
        tx = t+dt*x
        ty = t+dt*(1-y)
        fx = f(tx)
        fy = f(ty)
        cx, sx = real(fx), imag(fx)
        cy, sy = real(fy), imag(fy)
        c += w*(cy-cx)
        s += w*(sy-sx)
        r += w*(cy*sx-cx*sy)
    end
    c *= 0.25*dt^2
    s *= 0.25*dt^2
    r *= 0.25*dt^2
    (c, s, r)
end


function get_A(H::Hubbard, t::Real, dt::Real; 
               compute_derivative::Bool=false, matrix_times_minus_i::Bool=true)
    if  compute_derivative
        error("compute_derivative not yet implemented")
    else
        fac_diag = dt 
        fac_offdiag = get_integral_f(t, dt, H.f)
    end
    
    HubbardState(matrix_times_minus_i, compute_derivative, fac_diag, fac_offdiag, H)
end


struct BState <: TimeDependentSchroedingerMatrixState
    matrix_times_minus_i :: Bool
    compute_derivative :: Bool
    H::Hubbard
    c::Float64
    s::Float64
    r::Float64
    Hdu::Array{Complex{Float64},1}
    Hsu::Array{Complex{Float64},1}
    Hau::Array{Complex{Float64},1}
    v::Array{Complex{Float64},1}
    w::Array{Complex{Float64},1}
end

function get_B(H::Hubbard, t::Real, dt::Real,
               h1::Array{Complex128,1},
               h2::Array{Complex128,1},
               h3::Array{Complex128,1},
               h4::Array{Complex128,1},
               h5::Array{Complex128,1};
               compute_derivative::Bool=false, matrix_times_minus_i::Bool=true)
    if  compute_derivative
        error("compute_derivative not yet implemented")
    else
        #r = get_integral_r(t, dt, H.f)
        #f = get_integral_f(t, dt, H.f, with_factor=true)
        #c, s = real(f), imag(f)
        c, s, r = get_integrals_csr(t, dt, H.f)
    end
    
    BState(matrix_times_minus_i, compute_derivative, H, c, s, r, h1, h2, h3, h4, h5)
end

import Base.LinAlg: A_mul_B!, issymmetric, ishermitian, checksquare
import Base: eltype, size, norm, full

size(B::BState) = size(B.H)
size(B::BState, dim::Int) = size(B.H, dim) 
eltype(B::BState) = eltype(B.H) 
issymmetric(B::BState) = issymmetric(B.H) # TODO: check 
ishermitian(B::BState) = ishermitian(B.H) # TODO: check 
checksquare(B::BState) = checksquare(B.H)


function full(B::BState) 
    return full((-1im*B.c)*(B.H.H_upper_symm*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_symm)+
    B.s*(B.H.H_upper_anti*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_anti)+
    B.r*(B.H.H_upper_symm*B.H.H_upper_anti-B.H.H_upper_anti*B.H.H_upper_symm))
end



function A_mul_B!(y, B::BState, u)
    B.Hdu[:] = B.H.H_diag.*u
    B.Hsu[:] = B.H.H_upper_symm*u
    B.Hau[:] = B.H.H_upper_anti*u
    if !B.H.store_full_matrices
        B.Hsu[:] += At_mul_B(B.H.H_upper_symm, u)
        B.Hau[:] -= At_mul_B(B.H.H_upper_anti, u)
    end
    B.v[:] = (-1im*B.c)*B.Hdu+B.r*B.Hau
    y[:] = B.H.H_upper_symm*B.v
    if !B.H.store_full_matrices
        y[:] += At_mul_B(B.H.H_upper_symm, B.v)
    end
    B.v[:] = B.s*B.Hdu-B.r*B.Hsu
    B.w[:] = B.H.H_upper_anti*B.v
    if !B.H.store_full_matrices
        B.w[:] -= At_mul_B(B.H.H_upper_anti, B.v)
    end    
    y[:] += B.w
    B.v[:] = (-1im)*B.c*B.Hsu+B.s*B.Hau
    B.w[:] = B.H.H_diag.*B.v
    y[:] -= B.w
end

abstract type MagnusStrang end


using FExpokit

import FExpokit: get_lwsp_liwsp_expv

function get_lwsp_liwsp_expv(H, scheme::Type{MagnusStrang}, m::Integer=30) 
    (lw, liw) = get_lwsp_liwsp_expv(size(H, 2), m)
    (lw+size(H, 2), liw)
end

get_order(::Type{MagnusStrang}) = 4
number_of_exponentials(::Type{MagnusStrang}) = 3


function TimeDependentLinearODESystems.step!(psi::Array{Complex{Float64},1}, H::Hubbard, 
               t::Real, dt::Real, scheme::Type{MagnusStrang},
               wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
               use_expm::Bool=false)
    global XA, XB
    h1 = similar(psi) # TODO: take somthing from wsp
    h2 = similar(psi) # TODO: take somthing from wsp
    h3 = similar(psi) # TODO: take somthing from wsp
    h4 = similar(psi) # TODO: take somthing from wsp
    h5 = similar(psi) # TODO: take somthing from wsp
    A = get_A(H, t, dt, matrix_times_minus_i=false)            
    B = get_B(H, t, dt, h1, h2, h3, h4, h5, matrix_times_minus_i=false)
    nA = H.norm0*dt
    nB = 0.5*H.norm0*dt^3
    if use_expm
        psi[:] = expm(-0.5im*full(B))*psi
        psi[:] = expm(-1im*full(A))*psi
        psi[:] = expm(-0.5im*full(B))*psi
        #psi[:] = expm(-1im*(full(A)+full(B)))*psi
    else
        expv!(psi, 0.5, B, psi, anorm=nB, 
             matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
        expv!(psi, 1.0, A, psi, anorm=nA, 
             matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
        expv!(psi, 0.5, B, psi, anorm=nB, 
             matrix_times_minus_i=true, hermitian=true, wsp=wsp, iwsp=iwsp)
    end
end  

