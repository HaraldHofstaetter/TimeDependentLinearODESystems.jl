load_example("hubbard.jl")

export MagnusStrang4

function get_integrals_csr(t::Real, dt::Real, f::Function)
    # order 4:
    #xyw=[(0.445948490915965, 0.445948490915965, 0.223381589678011),
    #     (0.445948490915965, 0.108103018168070, 0.223381589678011),
    #     (0.108103018168070, 0.445948490915965, 0.223381589678011),
    #     (0.091576213509771, 0.091576213509771, 0.109951743655322),
    #     (0.091576213509771, 0.816847572980459, 0.109951743655322),
    #     (0.816847572980459, 0.091576213509771, 0.109951743655322)]
    #  
    # order 6:
    xyw=[(0.333333333333333, 0.333333333333333, 0.225000000000000),
         (0.470142064105115, 0.470142064105115, 0.132394152788506),
         (0.470142064105115, 0.059715871789770, 0.132394152788506),
         (0.059715871789770, 0.470142064105115, 0.132394152788506),
         (0.101286507323456, 0.101286507323456, 0.125939180544827),
         (0.101286507323456, 0.797426985353087, 0.125939180544827),
         (0.797426985353087, 0.101286507323456, 0.125939180544827)]; 
    c = 0.0
    s = 0.0
    r = 0.0
    for (x,y0,w) in xyw
        y = 1-y0 # mirrored triangle
        tx = t+dt*x
        ty = t+dt*y
        fx = f(tx)
        fy = f(ty)
        cx, sx = real(fx), imag(fx)
        cy, sy = real(fy), imag(fy)
        c += w*(cy - cx)
        s += w*(sy - sx)
        r += w*(cy*sx - cx*sy)
    end
    c *= 0.25*dt
    s *= 0.25*dt
    r *= 0.25*dt
    (c, s, r)
end

function get_integrals_csr_d(t::Real, dt::Real, f::Function, fd::Function; symmetrized_defect::Bool=false)
    # order 4:
    #xyw=[(0.445948490915965, 0.445948490915965, 0.223381589678011),
    #     (0.445948490915965, 0.108103018168070, 0.223381589678011),
    #     (0.108103018168070, 0.445948490915965, 0.223381589678011),
    #     (0.091576213509771, 0.091576213509771, 0.109951743655322),
    #     (0.091576213509771, 0.816847572980459, 0.109951743655322),
    #     (0.816847572980459, 0.091576213509771, 0.109951743655322)]
    
    # order 6:
    xyw=[(0.333333333333333, 0.333333333333333, 0.225000000000000),
         (0.470142064105115, 0.470142064105115, 0.132394152788506),
         (0.470142064105115, 0.059715871789770, 0.132394152788506),
         (0.059715871789770, 0.470142064105115, 0.132394152788506),
         (0.101286507323456, 0.101286507323456, 0.125939180544827),
         (0.101286507323456, 0.797426985353087, 0.125939180544827),
         (0.797426985353087, 0.101286507323456, 0.125939180544827)]; 
    c = 0.0
    s = 0.0
    r = 0.0
    cd = 0.0
    sd = 0.0
    rd = 0.0
    for (x,y0,w) in xyw
        y = 1-y0 # mirrored triangle
        tx = t+dt*x
        ty = t+dt*y
        fx = f(tx)
        fy = f(ty)
        fdx = fd(tx)
        fdy = fd(ty)
        cx, sx = real(fx), imag(fx)
        cy, sy = real(fy), imag(fy)
        cdx, sdx = real(fdx), imag(fdx)
        cdy, sdy = real(fdy), imag(fdy)
        c += w*(cy - cx)
        s += w*(sy - sx)
        r += w*(cy*sx - cx*sy)
        if symmetrized_defect
            cd += w*(cdy*(y-0.5) - cdx*(x-0.5))
            sd += w*(sdy*(y-0.5) - sdx*(x-0.5))
            rd += w*(cdy*sx*(y-0.5) + cy*sdx*(x-0.5) - cdx*sy*(x-0.5) - cx*sdy*(y-0.5))
        else
            cd += w*(cdy*y - cdx*x)
            sd += w*(sdy*y - sdx*x)
            rd += w*(cdy*sx*y + cy*sdx*x - cdx*sy*x - cx*sdy*y)
        end
    end
    cd = 0.5*c + 0.25*dt*cd
    sd = 0.5*s + 0.25*dt*sd
    rd = 0.5*r + 0.25*dt*rd
    (cd, sd, rd)
end



struct BState <: TimeDependentSchroedingerMatrixState
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
               h1::Array{Complex{Float64},1},
               h2::Array{Complex{Float64},1},
               h3::Array{Complex{Float64},1},
               h4::Array{Complex{Float64},1},
               h5::Array{Complex{Float64},1};
               compute_derivative::Bool=false,
               symmetrized_defect::Bool=false)
    if compute_derivative
        c, s, r = get_integrals_csr_d(t, dt, H.f, H.fd, symmetrized_defect=symmetrized_defect)
    else
        c, s, r = get_integrals_csr(t, dt, H.f)
    end
    
    BState(H, c, s, r, h1, h2, h3, h4, h5)
end


LinearAlgebra.size(B::BState) = size(B.H)
LinearAlgebra.size(B::BState, dim::Int) = size(B.H, dim) 
LinearAlgebra.eltype(B::BState) = Complex{Float64}
LinearAlgebra.issymmetric(B::BState) = false
LinearAlgebra.ishermitian(B::BState) = true
LinearAlgebra.checksquare(B::BState) = B.H.N_psi


function full(B::BState) 
        full((-1im*B.c)*(B.H.H_upper_symm*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_symm)+
        B.s*(B.H.H_upper_anti*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_anti)+
        B.r*(B.H.H_upper_symm*B.H.H_upper_anti-B.H.H_upper_anti*B.H.H_upper_symm))
end



function LinearAlgebra.mul!(y, B::BState, u)
    B.Hdu[:] = B.H.H_diag.*u
    B.Hsu[:] = B.H.H_upper_symm*u
    B.Hau[:] = B.H.H_upper_anti*u
    if B.H.store_upper_part_only
        B.Hsu[:] += B.H.H_upper_symm'*u
        B.Hau[:] -= B.H.H_upper_anti'*u
    end
    B.v[:] = (-1im*B.c)*B.Hdu+B.r*B.Hau
    y[:] = B.H.H_upper_symm*B.v
    if B.H.store_upper_part_only
        y[:] += B.H.H_upper_symm'*B.v
    end
    B.v[:] = B.s*B.Hdu-B.r*B.Hsu
    B.w[:] = B.H.H_upper_anti*B.v
    if B.H.store_upper_part_only
        B.w[:] -= B.H.H_upper_anti'*B.v
    end    
    y[:] += B.w
    B.v[:] = (-1im)*B.c*B.Hsu+B.s*B.Hau
    B.w[:] = B.H.H_diag.*B.v
    y[:] -= B.w
    B.H.counter += 1
end

mutable struct MagnusStrang <: Scheme 
    p::Int
    symmetrized_defect::Bool
    function MagnusStrang(
            p::Int;
            symmetrized_defect::Bool=false,
            )
        new(p,
        symmetrized_defect)
    end
end

function (M::MagnusStrang)(;
    symmetrized_defect::Bool=M.symmetrized_defect)
    M.symmetrized_defect=symmetrized_defect
    M
end

MagnusStrang4 = MagnusStrang(4)

get_lwsp(H, scheme::MagnusStrang, m) = m+8
get_order(scheme::MagnusStrang) = scheme.p
number_of_exponentials(::MagnusStrang) = 3


function TimeDependentLinearODESystems.step!(psi::Array{Complex{Float64},1}, H::Hubbard, 
               t::Real, dt::Real, scheme::MagnusStrang,
               wsp::Vector{Vector{Complex{Float64}}};
               expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    h1 = wsp[expmv_m+3]
    h2 = wsp[expmv_m+4]
    h3 = wsp[expmv_m+5]
    h4 = wsp[expmv_m+6]
    h5 = wsp[expmv_m+7]
    #order 2
    #x = [1/2]
    #w = [1.0]
    #order 4
    #x = [1/2-sqrt(1/12), 1/2+sqrt(1/12)]
    #w = [1/2, 1/2]
    #order 6
    x = [1/2-sqrt(3/20), 1/2, 1/2+sqrt(3/20)] 
    w = [5/18, 8/18, 5/18] 

    tt = t+dt*x

    A = H(tt, w)
    B = get_B(H, t, dt, h1, h2, h3, h4, h5)
    expmv1!(psi, 0.5*dt, B, psi, expmv_tol, expmv_m, wsp) 
    expmv1!(psi,     dt, A, psi, expmv_tol, expmv_m, wsp)
    expmv1!(psi, 0.5*dt, B, psi, expmv_tol, expmv_m, wsp)
end  


function TimeDependentLinearODESystems.step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::Hubbard, 
                 t::Real, dt::Real,
                 scheme::MagnusStrang,
                 wsp::Vector{Vector{Complex{Float64}}};
                 expmv_tol::Real=1e-1, expmv_m::Int=min(30, size(H,1)))
    h = wsp[expmv_m+8]
    h1 = wsp[expmv_m+3]
    h2 = wsp[expmv_m+4]
    h3 = wsp[expmv_m+5]
    h4 = wsp[expmv_m+6]
    h5 = wsp[expmv_m+7]
    #order 2
    #x = [1/2]
    #w = [1.0]
    #order 4
    #x = [1/2-sqrt(1/12), 1/2+sqrt(1/12)]
    #w = [1/2, 1/2]
    #order 6
    x = [1/2-sqrt(3/20), 1/2, 1/2+sqrt(3/20)] 
    w = [5/18, 8/18, 5/18] 

    tt = t .+ dt*x

    if scheme.symmetrized_defect
        H1 = H(t)
        mul1!(psi_est, H1, psi)
        psi_est[:] *= -0.5
    else
        psi_est[:] .= 0.0
    end

    #1/2 B -----------------------

    B = get_B(H, t, dt, h1, h2, h3, h4, h5)
    expmv1!(psi, 0.5*dt, B, psi, expmv_tol, expmv_m, wsp)
    if scheme.symmetrized_defect
        expmv1!(psi_est, 0.5*dt, B, psi_est, expmv_tol, expmv_m, wsp)
    end
    G = get_B(H, t, dt, h1, h2, h3, h4, h5,
              compute_derivative=true, symmetrized_defect=scheme.symmetrized_defect)
    mul1!(h, G, psi) 
    psi_est[:] += 0.5*dt*h[:]

    #A -----------------------

    A = H(tt, w)
    expmv1!(psi, dt, A, psi, expmv_tol, expmv_m, wsp)
    expmv1!(psi_est, dt, A, psi_est, expmv_tol, expmv_m, wsp)

    A = H(tt, w)
    if scheme.symmetrized_defect
        Ad = H(tt, w.*(x .- 0.5), compute_derivative=true)
    else
        Ad = H(tt, w.*x, compute_derivative=true)
    end
    Gamma!(h, A, Ad, psi, 4, dt, h1, h2, h3, h4) 
    psi_est[:] += h[:]

    #1/2 B -----------------------

    expmv1!(psi, 0.5*dt, B, psi, expmv_tol, expmv_m, wsp)
    expmv1!(psi_est, 0.5*dt, B, psi_est, expmv_tol, expmv_m, wsp)
    G = get_B(H, t, dt, h1, h2, h3, h4, h5, 
              compute_derivative=true, symmetrized_defect=scheme.symmetrized_defect)
    mul1!(h, G, psi)
    psi_est[:] += 0.5*dt*h[:]

    #-----------------------------

    H1 = H(t+dt)
    mul1!(h, H1, psi)
    if scheme.symmetrized_defect
        h[:] *= 0.5
    end
    psi_est[:] -= h[:]

    psi_est[:] *= dt/5
end


