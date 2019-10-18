load_example("hubbard.jl")

export MagnusStrang

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
    matrix_times_minus_i :: Bool
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
               compute_derivative::Bool=false, matrix_times_minus_i::Bool=true,
               symmetrized_defect::Bool=false)
    if compute_derivative
        c, s, r = get_integrals_csr_d(t, dt, H.f, H.fd, symmetrized_defect=symmetrized_defect)
    else
        c, s, r = get_integrals_csr(t, dt, H.f)
    end
    
    BState(matrix_times_minus_i, H, c, s, r, h1, h2, h3, h4, h5)
end


LinearAlgebra.size(B::BState) = size(B.H)
LinearAlgebra.size(B::BState, dim::Int) = size(B.H, dim) 
LinearAlgebra.eltype(B::BState) = Complex{Float64}
LinearAlgebra.issymmetric(B::BState) = false
LinearAlgebra.ishermitian(B::BState) = !B.matrix_times_minus_i 
LinearAlgebra.checksquare(B::BState) = B.H.N_psi


function full(B::BState) 
    if B.matrix_times_minus_i
        return full(-(B.H.H_upper_symm*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_symm)+
                    (-1im*B.s)*(B.H.H_upper_anti*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_anti)+
                    (-1im*B.r)*(B.H.H_upper_symm*B.H.H_upper_anti-B.H.H_upper_anti*B.H.H_upper_symm))
    else
        return full((-1im*B.c)*(B.H.H_upper_symm*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_symm)+
        B.s*(B.H.H_upper_anti*diagm(B.H.H_diag)-diagm(B.H.H_diag)*B.H.H_upper_anti)+
        B.r*(B.H.H_upper_symm*B.H.H_upper_anti-B.H.H_upper_anti*B.H.H_upper_symm))
    end
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
    if B.matrix_times_minus_i
        y[:] *= -1im
    end
end

abstract type MagnusStrang end


function get_lwsp_liwsp_expv(H, scheme::Type{MagnusStrang}, m::Integer=30) 
    (lw, liw) = get_lwsp_liwsp_expv(size(H, 2), m)
    (lw+size(H, 2), liw)
end

get_order(::Type{MagnusStrang}) = 4
number_of_exponentials(::Type{MagnusStrang}) = 3


function TimeDependentLinearODESystems.step!(psi::Array{Complex{Float64},1}, H::Hubbard, 
               t::Real, dt::Real, scheme::Type{MagnusStrang},
               wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
               expmv_tol::Real=1e-7)
    h1 = similar(psi) # TODO: take somthing from wsp
    h2 = similar(psi) # TODO: take somthing from wsp
    h3 = similar(psi) # TODO: take somthing from wsp
    h4 = similar(psi) # TODO: take somthing from wsp
    h5 = similar(psi) # TODO: take somthing from wsp
    #order 2
    #x = [1/2]
    #w = [1.0]
    #order 4
    x = [1/2-sqrt(1/12), 1/2+sqrt(1/12)]
    w = [1/2, 1/2]
    #order 6
    #x = [1/2-sqrt(3/20), 1/2, 1/2+sqrt(3/20)] 
    #w = [5/18, 8/18, 5/18] 

    tt = t+dt*x

    A = H(tt, w, matrix_times_minus_i=false)
    B = get_B(H, t, dt, h1, h2, h3, h4, h5, matrix_times_minus_i=false)
    if expmv_tol==0
        psi[:] = exp(-0.5im*dt*full(B))*psi
        psi[:] = exp(-1im*dt*full(A))*psi
        psi[:] = exp(-0.5im*dt*full(B))*psi
        #psi[:] = exp(-1im*dt*(full(A)+full(B)))*psi
    else
        expmv!(psi, -0.5im*dt, B, psi, tol=expmv_tol) 
        expmv!(psi,   -1im*dt, A, psi, tol=expmv_tol)
        expmv!(psi, -0.5im*dt, B, psi, tol=expmv_tol)
    end
end  


function TimeDependentLinearODESystems.step_estimated!(psi::Array{Complex{Float64},1}, psi_est::Array{Complex{Float64},1},
                 H::Hubbard, 
                 t::Real, dt::Real,
                 scheme::Type{MagnusStrang},
                 wsp::Array{Complex{Float64},1}, iwsp::Array{Int32,1};
                 symmetrized_defect::Bool=false, 
                 trapezoidal_rule::Bool=false, 
                 modified_Gamma::Bool=false,
                 expmv_tol::Real=1e-1)
    h = similar(psi) # TODO: take somthing from wsp
    h1 = similar(psi) # TODO: take somthing from wsp
    h2 = similar(psi) # TODO: take somthing from wsp
    h3 = similar(psi) # TODO: take somthing from wsp
    h4 = similar(psi) # TODO: take somthing from wsp
    h5 = similar(psi) # TODO: take somthing from wsp
    #order 2
    #x = [1/2]
    #w = [1.0]
    #order 4
    x = [1/2-sqrt(1/12), 1/2+sqrt(1/12)]
    w = [1/2, 1/2]
    #order 6
    #x = [1/2-sqrt(3/20), 1/2, 1/2+sqrt(3/20)] 
    #w = [5/18, 8/18, 5/18] 

    tt = t .+ dt*x

    if symmetrized_defect
        H1 = H(t, matrix_times_minus_i=true)
        mul!(psi_est, H1, psi)
        psi_est[:] *= -0.5
    else
        psi_est[:] .= 0.0
    end

    #1/2 B -----------------------

    B = get_B(H, t, dt, h1, h2, h3, h4, h5, matrix_times_minus_i=false)
    if expmv_tol==0
        psi[:] =         exp(-0.5im*dt*full(B))*psi
        if symmetrized_defect
            psi_est[:] = exp(-0.5im*dt*full(B))*psi_est
        end
    else
        expmv!(psi, -0.5im*dt, B, psi, tol=expmv_tol)
        if symmetrized_defect
            expmv!(psi_est, -0.5im*dt, B, psi_est, tol=expmv_tol)
        end
    end
    G = get_B(H, t, dt, h1, h2, h3, h4, h5, matrix_times_minus_i=true,
              compute_derivative=true, symmetrized_defect=symmetrized_defect)
    mul!(h, G, psi)
    psi_est[:] += 0.5*dt*h[:]

    #A -----------------------

    A = H(tt, w, matrix_times_minus_i=false)
    if expmv_tol==0
        psi[:]     = exp(-1im*dt*full(A))*psi
        psi_est[:] = exp(-1im*dt*full(A))*psi_est
    else
        expmv!(psi, -1im*dt, A, psi, tol=expmv_tol)
        expmv!(psi_est, -1im*dt, A, psi_est, tol=expmv_tol)
    end

    A = H(tt, w, matrix_times_minus_i=true)
    if symmetrized_defect
        Ad = H(tt, w.*(x .- 0.5), compute_derivative=true, matrix_times_minus_i=true)
    else
        Ad = H(tt, w.*x, compute_derivative=true, matrix_times_minus_i=true)
    end
    Gamma!(h, A, Ad, psi, 4, dt, h1, h2, h3, h4) 
    psi_est[:] += h[:]

    #1/2 B -----------------------

    if expmv_tol==0
        psi[:] =     exp(-0.5im*dt*full(B))*psi
        psi_est[:] = exp(-0.5im*dt*full(B))*psi_est
    else
        expmv!(psi, -0.5im*dt, B, psi, tol=expmv_tol)
        expmv!(psi_est, -0.5im*dt, B, psi_est, tol=expmv_tol)
    end
    G = get_B(H, t, dt, h1, h2, h3, h4, h5, matrix_times_minus_i=true, 
              compute_derivative=true, symmetrized_defect=symmetrized_defect)
    mul!(h, G, psi)
    psi_est[:] += 0.5*dt*h[:]

    #-----------------------------

    H1 = H(t+dt, matrix_times_minus_i=true)
    mul!(h, H1, psi)
    if symmetrized_defect
        h[:] *= 0.5
    end
    psi_est[:] -= h[:]

    psi_est[:] *= dt/5
end


