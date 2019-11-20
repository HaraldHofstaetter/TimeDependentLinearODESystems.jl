load_example("hubbard.jl")

export CF2g4BF, CF4g6BF, CF4oBF, CF4oHBF, gen_CF4

function mul_diag!(Y, H::Hubbard, B)
    Y[:] = H.H_diag.*B
end

function mul_symm!(Y, H::Hubbard, B)
    Y[:] = H.H_upper_symm*B 
    if H.store_upper_part_only
        Y[:] +=  H.H_upper_symm'*B
    end
end

function mul_anti!(Y, H::Hubbard, B)
    Y[:] = H.H_upper_anti*B 
    if H.store_upper_part_only
        Y[:] -=  H.H_upper_anti'*B
    end
end



mutable struct SchemeWithBruteForceErrorEstimator <: Scheme 
    scheme :: Scheme
    T::Matrix{Float64}
    CL::Vector{Float64}  #coeffs of [A1,[A1,[A1,A2]]], [A2,[A1,A2]], 
                         #[A1,[A1,A3]], [A2,A3] in leading term of local error
    exponentiate::Bool
    symmetrized::Bool
end

CF2g4BF = SchemeWithBruteForceErrorEstimator(CF2g4,
        [1/2        1/2
         sqrt(3)/2 sqrt(3)/2],
        [1/6], false, false) 

CF4g6BF = SchemeWithBruteForceErrorEstimator(CF4g6,
        [ 5/18        4/9   5/18
        -sqrt(15)/6   0    sqrt(15)/6
          5/9       -10/9    5/9],
        [1/1440, -1/540, -1/60, 1/30], false, false) 

CF4oBF = SchemeWithBruteForceErrorEstimator(CF4o,
        [ 5/18        4/9   5/18
        -sqrt(15)/6   0    sqrt(15)/6
          5/9       -10/9    5/9],
        [-1/115200, -31/454140, 1/4000, 1/870], false, false) 

CF4oHBF = SchemeWithBruteForceErrorEstimator(CF4oH,
        [ 5/18        4/9   5/18
        -sqrt(15)/6   0    sqrt(15)/6
          5/9       -10/9    5/9],
        [-8.544743700441166636878456E-7, -0.819876284228042871927452461E-4, 
        5.08679647548227151745526E-8,  0.15148309849837715519531315151E-2], false, false)

get_lwsp(H, scheme::SchemeWithBruteForceErrorEstimator, m) = 
    get_lwsp(H, scheme.scheme, m)+8
#    max(get_lwsp(H, scheme.scheme, m), 8)

get_order(scheme::SchemeWithBruteForceErrorEstimator) = 
    get_order(scheme.scheme) 
number_of_exponentials(scheme::SchemeWithBruteForceErrorEstimator) = 
    number_of_exponentials(scheme.scheme)


legendre(n::Integer, x::T) where T = (-1)^n*sum([binomial(n,k)*binomial(n+k,k)*(-1)^k*(k==0 ? 1 : x^k) for k=0:n])

function gen_CF4(f21, f23, x::Vector, w::Vector; brute_force_err_est::Bool=false)
    q = length(x)
    @assert q==length(w)
    F=[ (1-f21)/2 -1/(3*f21+3)  -f23/2
        f21        0             f23
        (1-f21)/2  1/(3*f21+3)  -f23/2]

    T = [(2*n-1)*w[m]*legendre(n-1,x[m]) for n=1:3, m=1:q]
    A = F*T
    CF4 = TimeDependentLinearODESystems.CommutatorFreeScheme(A, x, 4)

    if brute_force_err_est
        CL = [-(1/288)*f21^2+1/1440,  ((1/60)*f21^2-(1/270)*f21-1/540)*(1/(f21+1)^2),
              -1/60-(1/24)*f23-(1/24)*f21*f23, ((1/30)*f21+1/30+(1/6)*f23)*(1/(f21+1))]
        CF4BF = SchemeWithBruteForceErrorEstimator(CF4, T, CL, false)
        return CF4BF
    else
        return CF4
    end
end


function TimeDependentLinearODESystems.step_estimated!(
             psi::Array{Complex{Float64},1}, 
             psi_est::Array{Complex{Float64},1},
             H::Hubbard, 
             t::Real, dt::Real,
             scheme::SchemeWithBruteForceErrorEstimator,
             wsp::Vector{Vector{Complex{Float64}}};
             expmv_tol::Real=1e-7, expmv_m::Int=min(30, size(H,1)))
    p = get_order(scheme)
    @assert p==2||p==4 "Brute force error estimator implemented for order 4 only"

    ff = H.f.(t .+ dt*scheme.scheme.c)
    f = scheme.T * ff

    if scheme.exponentiate
        wsp1 = unsafe_wrap(Vector{Vector{Complex{Float64}}}, 
                           pointer(wsp,get_lwsp(H, scheme, expmv_m)-7),8)    
        fac_B = scheme.symmetrized ? -0.5im : -1im
        if scheme.symmetrized
            if p==2
                B = BF2State(H, -1im*dt, fac_B, f, scheme.CL, wsp1) 
            elseif p==4
                B = BF4State(H, -1im*dt, fac_B, f, scheme.CL, wsp1) 
            end
            expmv1!(psi, 1, B, psi, expmv_tol, 8, wsp) # first argument psi NOT pis_est! CHECK m=8 !!!
        end
    else
        copyto!(psi_est, psi)
    end

    step!(psi, H, t, dt, scheme.scheme, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
    
    if scheme.exponentiate
        wsp1 = unsafe_wrap(Vector{Vector{Complex{Float64}}}, 
                           pointer(wsp,get_lwsp(H, scheme, expmv_m)-7),8)    
        fac_B = scheme.symmetrized ? -0.5im : -1im
        if p==2
            B = BF2State(H, -1im*dt, fac_B, f, scheme.CL, wsp1) 
        elseif p==4
            B = BF4State(H, -1im*dt, fac_B, f, scheme.CL, wsp1) 
        end
        expmv1!(psi_est, 1, B, psi, expmv_tol, 8, wsp) # CHECK m=8 !!!
        psi_est[:] = psi[:] - psi_est[:]
    else
        if p==2
            B = BF2State(H, -1im*dt, 1, f, scheme.CL, wsp) 
        elseif p==4
            B = BF4State(H, -1im*dt, 1, f, scheme.CL, wsp) 
        end
        mul!(psi_est, B, psi_est)
    end
end


struct BF2State <: TimeDependentSchroedingerMatrixState
    H::Hubbard
    fac1::Complex{Float64}
    fac2::Complex{Float64}
    f::Vector{Complex{Float64}}
    CL::Vector{Float64}
    wsp::Vector{Vector{Complex{Float64}}}
end


LinearAlgebra.size(B::BF2State) = size(B.H)
LinearAlgebra.size(B::BF2State, dim::Int) = size(B.H, dim) 
LinearAlgebra.eltype(B::BF2State) = Complex{Float64}
LinearAlgebra.issymmetric(B::BF2State) = false
LinearAlgebra.ishermitian(B::BF2State) = true
LinearAlgebra.checksquare(B::BF2State) = B.H.N_psi


function LinearAlgebra.mul!(y, B::BF2State, u)             
    u1 = wsp[1]
    u2 = wsp[2] 
    E1 = wsp[3]

    fd1 = B.fac1
    fs1 = B.fac1*real(B.f[1])
    fa1 = 1im*B.fac1*imag(B.f[1])
    fs2 = B.fac1*real(B.f[2])
    fa2 = 1im*B.fac1*imag(B.f[2])
    fs3 = B.fac1*real(B.f[3])
    fa3 = 1im*B.fac1*imag(B.f[3])

    E1[:] .= 0

    H = B.H
    
    #u1 = H0*u
    mul_diag!(u1, H, u)
    
    #u2 = H0*H0*u1
    #mul_diag!(u2, H, u1)
    
    #u2 = H1*H0*u1
    mul_symm!(u2, H, u1)
    E1[:] += (-fs2*fd1)*u2
    
    #u2 = H2*H0*u1
    mul_anti!(u2, H, u1)
    E1[:] += (-fa2*fd1)*u2
    
    #u1 = H1*u
    mul_symm!(u1, H, u)
    
    #u2 = H0*H1*u1
    mul_diag!(u2, H, u1)
    E1[:] += (fd1*fs2)*u2
    
    #u2 = H1*H1*u1
    #mul_symm!(u2, H, u1)
    
    #u2 = H2*H1*u1
    mul_anti!(u2, H, u1)
    E1[:] += (fa1*fs2-fa2*fs1)*u2
    
    #u1 = H2*u
    mul_anti!(u1, H, u)
    
    #u2 = H0*H2*u1
    mul_diag!(u2, H, u1)
    E1[:] += (fd1*fa2)*u2
    
    #u2 = H1*H2*u1
    mul_symm!(u2, H, u1)
    E1[:] += (fs1*fa2-fs2*fa1)*u2
    
    #u2 = H2*H2*u1
    #mul_anti!(u2, H, u1)

    H.counter += 3 # (#mul_symm! + #mul_anti!)/2 

    y[:] = (B.fac2*B.CL[1])*E1
end


# Code of mul!(y, B::BF4State, u) generated  by the following Maple code:
# (for Expocon see https://github.com/HaraldHofstaetter/Expocon.mpl)

# with(Physics);
# with(Expocon);
# Setup(noncommutativeprefix = {H});
# 
# A1 := fd1*H[0]+fs1*H[1]+fa1*H[2]; 
# A2 := fs2*H[1]+fa2*H[2]; 
# A3 := fs3*H[1]+fa3*H[2];
# 
# C := Physics[Commutator];
# Q := ["diag", "symm", "anti"];
# 
# Z1 := C(A1, C(A1, C(A1, A2))); 
# Z2 := C(A2, C(A1, A2)); 
# Z3 := C(A1, C(A1, A3)); 
# Z4 := C(A2, A3); 
# 
# for j1 from 0 to 2 do 
#     printf("\n#u1 = H%a*u\n", j1); 
#     printf("mul_%s!(u1, H, u)\n", Q[j1+1]); 
#     for j2 from 0 to 2 do 
#         printf("\n# u2 = H%a*H%a*u\n", j2, j1); 
#         printf("mul_%s!(u2, H, u1)\n", Q[j2+1]); 
#         x := simplify(wcoeff([H[j2], H[j1]], Z4)); 
#         if x <> 0 then printf("E4[:] += (%a)*u2\n", x) end if; 
#         for j3 from 0 to 2 do printf("\n# u3 = H%a*H%a*H%a*u\n", j3, j2, j1); 
#             printf("mul_%s!(u3, H, u2)\n", Q[j3+1]); 
#             x := simplify(wcoeff([H[j3], H[j2], H[j1]], Z2)); 
#             if x <> 0 then printf("E2[:] += (%a)*u3\n", x) end if; 
#             x := simplify(wcoeff([H[j3], H[j2], H[j1]], Z3)); 
#             if x <> 0 then printf("E3[:] += (%a)*u3\n", x) end if; 
#             for j4 from 0 to 2 do printf("\n# u4 = H%a*H%a*H%a*H%a*u\n", j4, j3, j2, j1); 
#                 printf("mul_%s!(u4, H, u3)\n", Q[j4+1]); 
#                 x := simplify(wcoeff([H[j4], H[j3], H[j2], H[j1]], Z1)); 
#                 if x <> 0 then printf("E1[:] += (%a)*u4\n", x) end if 
#             end do 
#         end do 
#     end do 
# end do


struct BF4State <: TimeDependentSchroedingerMatrixState
    H::Hubbard
    fac1::Complex{Float64}
    fac2::Complex{Float64}
    f::Vector{Complex{Float64}}
    CL::Vector{Float64}
    wsp::Vector{Vector{Complex{Float64}}}
end


LinearAlgebra.size(B::BF4State) = size(B.H)
LinearAlgebra.size(B::BF4State, dim::Int) = size(B.H, dim) 
LinearAlgebra.eltype(B::BF4State) = Complex{Float64}
LinearAlgebra.issymmetric(B::BF4State) = false
LinearAlgebra.ishermitian(B::BF4State) = true
LinearAlgebra.checksquare(B::BF4State) = B.H.N_psi


function LinearAlgebra.mul!(y, B::BF4State, u)    
    #print("M ")
    u1 = B.wsp[1]
    u2 = B.wsp[2] 
    u3 = B.wsp[3] 
    u4 = B.wsp[4] 
    E1 = B.wsp[5]
    E2 = B.wsp[6]
    E3 = B.wsp[7] 
    E4 = B.wsp[8] 

    fd1 = B.fac1
    fs1 = B.fac1*real(B.f[1])
    fa1 = 1im*B.fac1*imag(B.f[1])
    fs2 = B.fac1*real(B.f[2])
    fa2 = 1im*B.fac1*imag(B.f[2])
    fs3 = B.fac1*real(B.f[3])
    fa3 = 1im*B.fac1*imag(B.f[3])

    E1[:] .= 0
    E2[:] .= 0
    E3[:] .= 0
    E4[:] .= 0

    H = B.H
    
    #u1 = H0*u
    mul_diag!(u1, H, u)
    
    # u2 = H0*H0*u
    mul_diag!(u2, H, u1)
    
    # u3 = H0*H0*H0*u
    mul_diag!(u3, H, u2)
    
    # u4 = H0*H0*H0*H0*u
    # mul_diag!(u4, H, u3)
    
    # u4 = H1*H0*H0*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fd1^3*fs2)*u4
    
    # u4 = H2*H0*H0*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (-fd1^3*fa2)*u4
    
    # u3 = H1*H0*H0*u
    mul_symm!(u3, H, u2)
    E3[:] += (fd1^2*fs3)*u3
    
    # u4 = H0*H1*H0*H0*u
    mul_diag!(u4, H, u3)
    E1[:] += (3*fd1^3*fs2)*u4
    
    # u4 = H1*H1*H0*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (2*fd1^2*fs1*fs2)*u4
    
    # u4 = H2*H1*H0*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (3*fa1*fd1^2*fs2-fa2*fd1^2*fs1)*u4
    
    # u3 = H2*H0*H0*u
    mul_anti!(u3, H, u2)
    E3[:] += (fd1^2*fa3)*u3
    
    # u4 = H0*H2*H0*H0*u
    mul_diag!(u4, H, u3)
    E1[:] += (3*fd1^3*fa2)*u4
    
    # u4 = H1*H2*H0*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fa1*fd1^2*fs2+3*fa2*fd1^2*fs1)*u4
    
    # u4 = H2*H2*H0*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (2*fa1*fd1^2*fa2)*u4
    
    # u2 = H1*H0*u
    mul_symm!(u2, H, u1)
    
    # u3 = H0*H1*H0*u
    mul_diag!(u3, H, u2)
    E3[:] += (-2*fd1^2*fs3)*u3
    
    # u4 = H0*H0*H1*H0*u
    mul_diag!(u4, H, u3)
    E1[:] += (-3*fd1^3*fs2)*u4
    
    # u4 = H1*H0*H1*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (-4*fd1^2*fs1*fs2)*u4
    
    # u4 = H2*H0*H1*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (-3*fa1*fd1^2*fs2-fa2*fd1^2*fs1)*u4
    
    # u3 = H1*H1*H0*u
    mul_symm!(u3, H, u2)
    E2[:] += (-fs2^2*fd1)*u3
    E3[:] += (-fs1*fd1*fs3)*u3
    
    # u4 = H0*H1*H1*H0*u
    #mul_diag!(u4, H, u3)
    
    # u4 = H1*H1*H1*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fs1^2*fd1*fs2)*u4
    
    # u4 = H2*H1*H1*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (-fa2*fd1*fs1^2)*u4
    
    # u3 = H2*H1*H0*u
    mul_anti!(u3, H, u2)
    E2[:] += (-fa2*fd1*fs2)*u3
    E3[:] += (-2*fa1*fd1*fs3+fa3*fd1*fs1)*u3
    
    # u4 = H0*H2*H1*H0*u
    mul_diag!(u4, H, u3)
    E1[:] += (-3*fd1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H1*H2*H1*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (-4*fa1*fd1*fs1*fs2+3*fa2*fd1*fs1^2)*u4
    
    # u4 = H2*H2*H1*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (-3*fa1^2*fd1*fs2+2*fa1*fa2*fd1*fs1)*u4
    
    # u2 = H2*H0*u
    mul_anti!(u2, H, u1)
    
    # u3 = H0*H2*H0*u
    mul_diag!(u3, H, u2)
    E3[:] += (-2*fd1^2*fa3)*u3
    
    # u4 = H0*H0*H2*H0*u
    mul_diag!(u4, H, u3)
    E1[:] += (-3*fd1^3*fa2)*u4
    
    # u4 = H1*H0*H2*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fa1*fd1^2*fs2-3*fa2*fd1^2*fs1)*u4
    
    # u4 = H2*H0*H2*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (-4*fa1*fd1^2*fa2)*u4
    
    # u3 = H1*H2*H0*u
    mul_symm!(u3, H, u2)
    E2[:] += (-fa2*fd1*fs2)*u3
    E3[:] += (fa1*fd1*fs3-2*fa3*fd1*fs1)*u3
    
    # u4 = H0*H1*H2*H0*u
    mul_diag!(u4, H, u3)
    E1[:] += (3*fd1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H1*H1*H2*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (2*fa1*fd1*fs1*fs2-3*fa2*fd1*fs1^2)*u4
    
    # u4 = H2*H1*H2*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (3*fa1^2*fd1*fs2-4*fa1*fa2*fd1*fs1)*u4
    
    # u3 = H2*H2*H0*u
    mul_anti!(u3, H, u2)
    E2[:] += (-fa2^2*fd1)*u3
    E3[:] += (-fa1*fd1*fa3)*u3
    
    # u4 = H0*H2*H2*H0*u
    # mul_diag!(u4, H, u3)
    
    # u4 = H1*H2*H2*H0*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fa1^2*fd1*fs2)*u4
    
    # u4 = H2*H2*H2*H0*u
    mul_anti!(u4, H, u3)
    E1[:] += (-fa1^2*fd1*fa2)*u4
    
    #u1 = H1*u
    mul_symm!(u1, H, u)
    
    # u2 = H0*H1*u
    mul_diag!(u2, H, u1)
    
    # u3 = H0*H0*H1*u
    mul_diag!(u3, H, u2)
    E3[:] += (fd1^2*fs3)*u3
    
    # u4 = H0*H0*H0*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (fd1^3*fs2)*u4
    
    # u4 = H1*H0*H0*H1*u
    #mul_symm!(u4, H, u3)
    
    # u4 = H2*H0*H0*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (fa1*fd1^2*fs2-fa2*fd1^2*fs1)*u4
    
    # u3 = H1*H0*H1*u
    mul_symm!(u3, H, u2)
    E2[:] += (2*fs2^2*fd1)*u3
    E3[:] += (2*fs1*fd1*fs3)*u3
    
    # u4 = H0*H1*H0*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (4*fd1^2*fs1*fs2)*u4
    
    # u4 = H1*H1*H0*H1*u
    mul_symm!(u4, H, u3)
    E1[:] += (3*fs1^2*fd1*fs2)*u4
    
    # u4 = H2*H1*H0*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (4*fa1*fd1*fs1*fs2-fa2*fd1*fs1^2)*u4
    
    # u3 = H2*H0*H1*u
    mul_anti!(u3, H, u2)
    E2[:] += (2*fa2*fd1*fs2)*u3
    E3[:] += (fa1*fd1*fs3+fa3*fd1*fs1)*u3
    
    # u4 = H0*H2*H0*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (fa1*fd1^2*fs2+3*fa2*fd1^2*fs1)*u4
    
    # u4 = H1*H2*H0*H1*u
    mul_symm!(u4, H, u3)
    E1[:] += (3*fa2*fd1*fs1^2)*u4
    
    # u4 = H2*H2*H0*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (fa1^2*fd1*fs2+2*fa1*fa2*fd1*fs1)*u4
    
    # u2 = H1*H1*u
    mul_symm!(u2, H, u1)
    
    # u3 = H0*H1*H1*u
    mul_diag!(u3, H, u2)
    E2[:] += (-fs2^2*fd1)*u3
    E3[:] += (-fs1*fd1*fs3)*u3
    
    # u4 = H0*H0*H1*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (-2*fd1^2*fs1*fs2)*u4
    
    # u4 = H1*H0*H1*H1*u
    mul_symm!(u4, H, u3)
    E1[:] += (-3*fs1^2*fd1*fs2)*u4
    
    # u4 = H2*H0*H1*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (-2*fa1*fd1*fs1*fs2-fa2*fd1*fs1^2)*u4
    
    # u3 = H1*H1*H1*u
    mul_symm!(u3, H, u2)
    
    # u4 = H0*H1*H1*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (fs1^2*fd1*fs2)*u4
    
    # u4 = H1*H1*H1*H1*u
    #mul_symm!(u4, H, u3)
    
    # u4 = H2*H1*H1*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (fs1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u3 = H2*H1*H1*u
    mul_anti!(u3, H, u2)
    E2[:] += (-fs2*(fa1*fs2-fa2*fs1))*u3
    E3[:] += (-fs1*(fa1*fs3-fa3*fs1))*u3
    
    # u4 = H0*H2*H1*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (-2*fa1*fd1*fs1*fs2+3*fa2*fd1*fs1^2)*u4
    
    # u4 = H1*H2*H1*H1*u
    mul_symm!(u4, H, u3)
    E1[:] += (-3*fs1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H2*H2*H1*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (-2*fa1*fs1*(fa1*fs2-fa2*fs1))*u4
    
    # u2 = H2*H1*u
    mul_anti!(u2, H, u1)
    E4[:] += (fa2*fs3-fa3*fs2)*u2
    
    # u3 = H0*H2*H1*u
    mul_diag!(u3, H, u2)
    E2[:] += (-fa2*fd1*fs2)*u3
    E3[:] += (fa1*fd1*fs3-2*fa3*fd1*fs1)*u3
    
    # u4 = H0*H0*H2*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (fa1*fd1^2*fs2-3*fa2*fd1^2*fs1)*u4
    
    # u4 = H1*H0*H2*H1*u
    mul_symm!(u4, H, u3)
    E1[:] += (-3*fa2*fd1*fs1^2)*u4
    
    # u4 = H2*H0*H2*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (fa1^2*fd1*fs2-4*fa1*fa2*fd1*fs1)*u4
    
    # u3 = H1*H2*H1*u
    mul_symm!(u3, H, u2)
    E2[:] += (2*fs2*(fa1*fs2-fa2*fs1))*u3
    E3[:] += (2*fs1*(fa1*fs3-fa3*fs1))*u3
    
    # u4 = H0*H1*H2*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (4*fa1*fd1*fs1*fs2-3*fa2*fd1*fs1^2)*u4
    
    # u4 = H1*H1*H2*H1*u
    mul_symm!(u4, H, u3)
    E1[:] += (3*fs1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H2*H1*H2*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (4*fa1*fs1*(fa1*fs2-fa2*fs1))*u4
    
    # u3 = H2*H2*H1*u
    mul_anti!(u3, H, u2)
    E2[:] += (fa2*(fa1*fs2-fa2*fs1))*u3
    E3[:] += (fa1*(fa1*fs3-fa3*fs1))*u3
    
    # u4 = H0*H2*H2*H1*u
    mul_diag!(u4, H, u3)
    E1[:] += (fa1^2*fd1*fs2)*u4
    
    # u4 = H1*H2*H2*H1*u
    #mul_symm!(u4, H, u3)
    
    # u4 = H2*H2*H2*H1*u
    mul_anti!(u4, H, u3)
    E1[:] += (fa1^2*(fa1*fs2-fa2*fs1))*u4
    
    #u1 = H2*u
    mul_anti!(u1, H, u)
    
    # u2 = H0*H2*u
    mul_diag!(u2, H, u1)
    
    # u3 = H0*H0*H2*u
    mul_diag!(u3, H, u2)
    E3[:] += (fd1^2*fa3)*u3
    
    # u4 = H0*H0*H0*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (fd1^3*fa2)*u4
    
    # u4 = H1*H0*H0*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fa1*fd1^2*fs2+fa2*fd1^2*fs1)*u4
    
    # u4 = H2*H0*H0*H2*u
    #mul_anti!(u4, H, u3)
    
    # u3 = H1*H0*H2*u
    mul_symm!(u3, H, u2)
    E2[:] += (2*fa2*fd1*fs2)*u3
    E3[:] += (fa1*fd1*fs3+fa3*fd1*fs1)*u3
    
    # u4 = H0*H1*H0*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (3*fa1*fd1^2*fs2+fa2*fd1^2*fs1)*u4
    
    # u4 = H1*H1*H0*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (2*fa1*fd1*fs1*fs2+fa2*fd1*fs1^2)*u4
    
    # u4 = H2*H1*H0*H2*u
    mul_anti!(u4, H, u3)
    E1[:] += (3*fa1^2*fd1*fs2)*u4
    
    # u3 = H2*H0*H2*u
    mul_anti!(u3, H, u2)
    E2[:] += (2*fa2^2*fd1)*u3
    E3[:] += (2*fa1*fd1*fa3)*u3
    
    # u4 = H0*H2*H0*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (4*fa1*fd1^2*fa2)*u4
    
    # u4 = H1*H2*H0*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fa1^2*fd1*fs2+4*fa1*fa2*fd1*fs1)*u4
    
    # u4 = H2*H2*H0*H2*u
    mul_anti!(u4, H, u3)
    E1[:] += (3*fa1^2*fd1*fa2)*u4
    
    # u2 = H1*H2*u
    mul_symm!(u2, H, u1)
    E4[:] += (-fa2*fs3+fa3*fs2)*u2
    
    # u3 = H0*H1*H2*u
    mul_diag!(u3, H, u2)
    E2[:] += (-fa2*fd1*fs2)*u3
    E3[:] += (-2*fa1*fd1*fs3+fa3*fd1*fs1)*u3
    
    # u4 = H0*H0*H1*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (-3*fa1*fd1^2*fs2+fa2*fd1^2*fs1)*u4
    
    # u4 = H1*H0*H1*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (-4*fa1*fd1*fs1*fs2+fa2*fd1*fs1^2)*u4
    
    # u4 = H2*H0*H1*H2*u
    mul_anti!(u4, H, u3)
    E1[:] += (-3*fa1^2*fd1*fs2)*u4
    
    # u3 = H1*H1*H2*u
    mul_symm!(u3, H, u2)
    E2[:] += (-fs2*(fa1*fs2-fa2*fs1))*u3
    E3[:] += (-fs1*(fa1*fs3-fa3*fs1))*u3
    
    # u4 = H0*H1*H1*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (fa2*fd1*fs1^2)*u4
    
    # u4 = H1*H1*H1*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fs1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H2*H1*H1*H2*u
    #mul_anti!(u4, H, u3)
    
    # u3 = H2*H1*H2*u
    mul_anti!(u3, H, u2)
    E2[:] += (-2*fa2*(fa1*fs2-fa2*fs1))*u3
    E3[:] += (-2*fa1*(fa1*fs3-fa3*fs1))*u3
    
    # u4 = H0*H2*H1*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (-3*fa1^2*fd1*fs2+4*fa1*fa2*fd1*fs1)*u4
    
    # u4 = H1*H2*H1*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (-4*fa1*fs1*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H2*H2*H1*H2*u
    mul_anti!(u4, H, u3)
    E1[:] += (-3*fa1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u2 = H2*H2*u
    mul_anti!(u2, H, u1)
    
    # u3 = H0*H2*H2*u
    mul_diag!(u3, H, u2)
    E2[:] += (-fa2^2*fd1)*u3
    E3[:] += (-fa1*fd1*fa3)*u3
    
    # u4 = H0*H0*H2*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (-2*fa1*fd1^2*fa2)*u4
    
    # u4 = H1*H0*H2*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fa1^2*fd1*fs2-2*fa1*fa2*fd1*fs1)*u4
    
    # u4 = H2*H0*H2*H2*u
    mul_anti!(u4, H, u3)
    E1[:] += (-3*fa1^2*fd1*fa2)*u4
    
    # u3 = H1*H2*H2*u
    mul_symm!(u3, H, u2)
    E2[:] += (fa2*(fa1*fs2-fa2*fs1))*u3
    E3[:] += (fa1*(fa1*fs3-fa3*fs1))*u3
    
    # u4 = H0*H1*H2*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (3*fa1^2*fd1*fs2-2*fa1*fa2*fd1*fs1)*u4
    
    # u4 = H1*H1*H2*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (2*fa1*fs1*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H2*H1*H2*H2*u
    mul_anti!(u4, H, u3)
    E1[:] += (3*fa1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u3 = H2*H2*H2*u
    mul_anti!(u3, H, u2)
    
    # u4 = H0*H2*H2*H2*u
    mul_diag!(u4, H, u3)
    E1[:] += (fa1^2*fd1*fa2)*u4
    
    # u4 = H1*H2*H2*H2*u
    mul_symm!(u4, H, u3)
    E1[:] += (-fa1^2*(fa1*fs2-fa2*fs1))*u4
    
    # u4 = H2*H2*H2*H2*u
    #mul_anti!(u4, H, u3)
    
    H.counter += 38 # (#mul_symm! + #mul_anti!)/2 
    
    y[:] = B.CL[1]*E1 
    y[:] += B.CL[2]*E2 
    y[:] += B.CL[3]*E3 
    y[:] += B.CL[4]*E4
    y[:] *= B.fac2
end
    
    
    
    
    
    
    
