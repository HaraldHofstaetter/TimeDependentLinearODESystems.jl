load_example("hubbard.jl")

export CF4BF, CF4g6BF, CF4oBF

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
    CL::Vector{Float64}  #coeffs of [A1,[A1,[A1,A2]]], [A2,[A1,A2]], 
                         #[A1,[A1,A3]], [A2,A3] in leading term of local error
end

CF4BF = SchemeWithBruteForceErrorEstimator(CF4,[1/1440, -1/540, -1/60, 1/30]) 
CF4g6BF = SchemeWithBruteForceErrorEstimator(CF4g6,[1/1440, -1/540, -1/60, 1/30]) 
CF4oBF = SchemeWithBruteForceErrorEstimator(CF4o,[-1/115200, -31/454140, 1/4000, 1/870]) 

get_lwsp(H, scheme::SchemeWithBruteForceErrorEstimator, m) = 
    max(get_lwsp(H, scheme.scheme, m), 8)
get_order(scheme::SchemeWithBruteForceErrorEstimator) = 
    get_order(scheme.scheme) 
number_of_exponentials(scheme::SchemeWithBruteForceErrorEstimator) = 
    number_of_exponentials(scheme.scheme)



function TimeDependentLinearODESystems.step_estimated!(
             psi::Array{Complex{Float64},1}, 
             psi_est::Array{Complex{Float64},1},
             H::Hubbard, 
             t::Real, dt::Real,
             scheme::SchemeWithBruteForceErrorEstimator,
             wsp::Vector{Vector{Complex{Float64}}};
             expmv_tol::Real=1e-1, expmv_m::Int=min(30, size(H,1)))
    @assert get_order(scheme)==4 "Brute force error estimator implemented for order 4 only"
    copyto!(psi_est, psi)
    step!(psi, H, t, dt, scheme.scheme, wsp, expmv_tol=expmv_tol, expmv_m=expmv_m)
    brute_force_error_estimator!(psi_est, H, t, dt, scheme.CL, wsp)
end


# Code of brute_force_error_estimator generated 
# by the following Maple code .
# For Expocon see https://github.com/HaraldHofstaetter/Expocon.mpl

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


function brute_force_error_estimator!( 
             psi_est::Array{Complex{Float64},1}, #inout, in: psi, out:: psi_est
             H::Hubbard, 
             t::Real, dt::Real,
             CL::Vector{Float64},
             wsp::Vector{Vector{Complex{Float64}}})
    u = psi_est
    u1 = wsp[1]
    u2 = wsp[2] 
    u3 = wsp[3] 
    u4 = wsp[4] 
    E1 = wsp[5]
    E2 = wsp[6]
    E3 = wsp[7] 
    E4 = wsp[8] 

    x = [1/2-sqrt(15)/10, 1/2, 1/2+sqrt(15)/10]
    ff = H.f.(t .+ dt*x)

    f1 = dt*(5/18*ff[1] + 4/9*ff[2] + 5/18*ff[3])
    f2 = dt*(-1/6*sqrt(15)*ff[1]    + 1/6*sqrt(15)*ff[3])
    f3 = dt*(5/9*ff[1]  -10/9*ff[2] + 5/9*ff[3])
    
    fd1 = -1im*dt
    fs1 = -1im*real(f1)
    fa1 = -1im*1im*imag(f1)
    fs2 = -1im*real(f2)
    fa2 = -1im*1im*imag(f2)
    fs3 = -1im*real(f3)
    fa3 = -1im*1im*imag(f3)

    E1[:] .= 0
    E2[:] .= 0
    E3[:] .= 0
    E4[:] .= 0
    
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
    

    H.counter += 38 # (#mul_symm! + #mul_anti!)/2 rounded up
    
    psi_est[:] = CL[1]*E1 
    psi_est[:] += CL[2]*E2 
    psi_est[:] += CL[3]*E3 
    psi_est[:] += CL[4]*E4

end
    
    
    
    
    
    
    
