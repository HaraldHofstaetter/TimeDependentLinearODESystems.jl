using LinearAlgebra

"""
`expmv(t, A, v; [tol=1e-7], [m=min(30, size(A,1))])`
 
Computes the matrix exponential acting on some vector, 
   `w=exp(t*A)*v`
for hermitian matrix `A`.  `A` can  be of any type that supports `size`, `ishermitian`, and `mul!`.
Note that `t` may be complex.

The calculation is based on Krylow approximation whose error is controlled by a
cheaply computable a posteriori error bound, see
> Tobias Jawecki, Winfried Auzinger, and Othmar Koch:
> Computable strict upper bounds for Krylov approximations to a class of matrix exponentials 
> and phi-functions, 2018, https://arxiv.org/abs/1809.03369.

This Julia implementation is based on a MATLAB code by Tobias Jawecki (tobias.jawecki@tuwien.ac.at),
some Julia-specific implementation techniques were inspired by the Julia package Expokit.jl 
(https://github.com/acroy/Expokit.jl ).

# Examples
```jldoctest
julia> using LinearAlgebra

julia> n = 10
10

julia> H = 1/4*SymTridiagonal(fill(2, n), fill(-1, n-1));

julia> v0 = Vector(1:n); v0 = v0/norm(v0)
10-element Array{Float64,1}:
 0.050964719143762556
 0.10192943828752511 
 0.15289415743128767 
 0.20385887657505022 
 0.2548235957188128  
 0.30578831486257535 
 0.3567530340063379  
 0.40771775315010045 
 0.458682472293863   
 0.5096471914376256  

julia> w = expmv(-5.0im, H, v0, tol=1e-8, m=10)
    
10-element Array{Complex{Float64},1}:
 0.05096390373695163 - 9.672241958644319e-7im 
 0.10192174939075013 + 6.175193282303271e-6im 
 0.15293630898846008 + 5.568419142668078e-5im 
  0.2042172968172267 - 0.0002514604632602682im
 0.25354515124721594 - 0.0020176393163153744im
 0.29606946164872827 + 0.005323733310184955im 
 0.37362888257144794 + 0.03883711468224804im  
  0.5304030354671269 - 0.03369015663540572im  
 0.45370068822075676 - 0.28062957944041544im  
   0.137546264581073 - 0.20702692836297326im  

```

"""
function expmv(t::Number, A, v::Vector{NT}; 
                tol::Real=1e-7, 
                m::Int=min(30, size(A,1))) where NT<:Number 
    if imag(t)!=0 || eltype(A)==Complex{real(eltype(A))}
        w = similar(v, Complex{real(NT)})
    else
        w = similar(v)
    end
    expmv!(w, t, A, v; tol=tol, m=m)
    return w
end

expmv!(t::Number, A, v::Vector{NT}; 
        tol::Real=1e-7, 
        m::Int=min(30, size(A,1))) where {NT<:Number} = expmv!(v, t, A, v; tol=tol, m=m)

function expmv!(w::Union{Vector{NT}, Vector{Complex{NT}}}, t::Number, A, v::Vector{NT}; 
                 tol::Real=1e-7, 
                 m::Int=min(30, size(A,1))) where NT<:Number 
    if size(v,1) != size(A,2)
        error("dimension mismatch")
    end
    if !LinearAlgebra.ishermitian(A)
        error("hermitian matrix expected")
    end
    t_out = abs(t)
    sig = t/t_out

    if imag(sig)==0
        sig = real(sig)
    end
    if (imag(sig)!=0 || eltype(v)==Complex{real(NT)} || eltype(A)==Complex{real(eltype(A))}) && 
        eltype(w)!= Complex{real(NT)}
        error("complex output array expected")
    end
    copyto!(w, v)

    z = similar(w)
    # storage for Krylov subspace vectors
    V = Array{typeof(w)}(undef,m+1)
    for k=1:m+1
        V[k] = similar(w)
    end
    T = SymTridiagonal(zeros(real(NT), m+1), zeros(real(NT), m))

    nstep = 0
    mb = m 
    early_break = false
    t_now = 0.0 

    maxsteps = 100
    zerotol = 1E-12

    while t_now < t_out
        nstep = nstep + 1
        t_fin = t_out - t_now

        beta = norm(w)
        rmul!(copyto!(V[1], w), 1/beta) #V[1] = (1/beta)*w
        gamfac = 1.0 
        tk = 1.0
  
        for k = 1:m # start Lanczos
            mul!(z, A, V[k])
            if k>=2
                axpy!(-T.ev[k-1], V[k-1], z)
            end

            T.dv[k] = real(dot(z, V[k]))
            axpy!(-T.dv[k], V[k], z)
            T.ev[k] = norm(z)

            # test if secondary diagonal entry is zero (lucky breakdown)
            if T.ev[k] < zerotol 
                mb = k
                t_step = t_fin
                early_break = true
                break
            end

            rmul!(copyto!(V[k+1], z), 1/T.ev[k]) #V[k+1] = (1.0/T[k+1,k])*z

            # test error estimate to stop Lanczos if approximation is already accurate enough
            gamfac *= T.ev[k]/k
            tk *= t_fin
            if  beta*gamfac*tk < t_fin*tol  # if '|v|*gamma_k*(dt)^k/k! < dt*tol' stop Lanczos after k many steps
                mb = k
                t_step = t_fin
                early_break = true
                break
            end

        end # Lanczos

        if !early_break # if Lanczos was not stopped, compute step size by error estimate
            t_step = min( (tol/(beta*gamfac))^(1//(m-1)), t_fin )
        end

        # compute exponential of small matrix T 
        E = exp((sig*t_step)*view(T,1:mb,1:mb))
        fill!(w, zero(NT))
        for k=1:mb
            axpy!(beta*E[k,1], V[k], w)
        end

        if nstep>maxsteps;
            @warn("Lanczos reached max number of discrete time steps")
            isuccess = 0
            break 
        end

        t_now +=  t_step
    end

    w 
end



    
