using LinearAlgebra
using SparseArrays
using Combinatorics
using Arpack
using Distributed

export Hubbard, HubbardState
export energy, groundstate, double_occupation, full

mutable struct Hubbard <: TimeDependentSchroedingerMatrix 
    N_s    :: Int
    n_up   :: Int
    n_down :: Int
    N_up   :: Int
    N_down :: Int
    N_psi  :: Int
    N_nz   :: Int

    v_symm      :: Array{Float64, 2}
    v_anti      :: Array{Float64, 2}
    U      :: Float64
    H_diag :: Array{Float64, 1}
    H_upper_symm ::  SparseMatrixCSC{Float64, Int}
    H_upper_anti ::  SparseMatrixCSC{Float64, Int}

    tab_up       :: Dict{BitArray{1},Int}
    tab_inv_up   :: Array{BitArray{1},1} 
    tab_down     :: Dict{BitArray{1},Int}
    tab_inv_down :: Array{BitArray{1},1}

    store_upper_part_only  :: Bool 

    f :: Function
    fd :: Function

    counter :: Int
end


mutable struct HubbardState <: TimeDependentSchroedingerMatrixState
    compute_derivative :: Bool
    fac_diag     :: Float64
    fac_offdiag  :: Complex{Float64}

    H::Hubbard
end

"""
Represents `H` evaluated at time `t`
"""
function (H::Hubbard)(t::Real; compute_derivative::Bool=false)
    if  compute_derivative
        fac_diag = 0.0
        fac_offdiag = H.fd(t)
    else
        fac_diag = 1.0
        fac_offdiag = H.f(t)
    end
    
    HubbardState(compute_derivative, fac_diag, fac_offdiag, H)
end

"""
Represents a linear combination `c[1] H(t[1]) + c[2] H(t[2]) + ...`

This is needed for the integration routine, where `c` are the weights
and `t` are the points needed in evaluation.
"""
function (H::Hubbard)(t::Vector{Float64}, c::Vector{Float64};
                      compute_derivative::Bool=false)
    n = length(t)
    @assert n==length(c)&&n>0 "t, c must be vectors of same length>1"
    if  compute_derivative
        fac_diag = 0.0
        fac_offdiag = sum([c[j]*H.fd(t[j]) for j=1:n])
    else
        fac_diag = sum(c) 
        fac_offdiag = sum([c[j]*H.f(t[j]) for j=1:n])
    end
    
    HubbardState(compute_derivative, fac_diag, fac_offdiag, H)
end


function comb_to_bitarray(N::Int, a::Array{Int,1})
    b = falses(N)
    for aa in a
        b[aa] = true
    end
    BitArray(b)   
end

bitarray_to_comb(psi::BitArray{1}) = [k for k=1:length(psi) if psi[k]]

function gen_tabs(N::Integer, n::Integer)
    tab = Dict{BitArray{1},Int}()
    psi0 = falses(N)
    tab_inv = [psi0 for k=1:binomial(N, n)]
    j=0
    for a in Combinatorics.combinations(1:N,n)
        j+=1
        psi = comb_to_bitarray(N, a)
        tab[psi] = j
        tab_inv[j] = psi
    end
    return tab, tab_inv
end

function differ_by_1_entry(psi::BitArray{1})
    N = length(psi)
    a1 = [k for k=1:N if psi[k]]
    a0 = [k for k=1:N if !psi[k]]
    n1 = length(a1)
    n0 = length(a0)
    psi0 = falses(N) 
    psi1_hops = [(psi0, (0,0)) for k=1:n0*n1]
    j = 0
    for i1=1:n1
        for i0=1:n0
            j+=1
            psi_new = copy(psi)
            psi_new[a0[i0]] = true
            psi_new[a1[i1]] = false            
            psi1_hops[j] = (psi_new, (a1[i1], a0[i0]))
        end
    end
    psi1_hops
end

function get_sign_up(psi_up::BitArray{1}, psi_down::BitArray{1}, hop::Tuple{Int, Int})
    a = minimum(hop)
    b = maximum(hop)
    s = sum(psi_up[a+1:b-1]) + sum(psi_down[a:b-1])
    isodd(s) ? -1 : +1
end

function get_sign_down(psi_up::BitArray{1}, psi_down::BitArray{1}, hop::Tuple{Int, Int})
    a = minimum(hop)
    b = maximum(hop)
    s = sum(psi_up[a+1:b]) + sum(psi_down[a+1:b-1])
    isodd(s) ? -1 : +1
end


function get_dims(H::Hubbard)
    i_up = 1
    psi_up = H.tab_inv_up[i_up]
    psi1_hops_up = differ_by_1_entry(psi_up)
    nn_down = zeros(Int, H.N_down)
    for i_down = 1:H.N_down
        psi_down = H.tab_inv_down[i_down]
        i = (i_up-1)*H.N_down + i_down
        for (psi_new, hop) in psi1_hops_up
            j = (H.tab_up[psi_new]-1)*H.N_down + i_down
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_down[i_down] += 1
            end
        end
        psi1_hops_down = differ_by_1_entry(psi_down)
        for (psi_new, hop) in psi1_hops_down
            j = (i_up-1)*H.N_down + H.tab_down[psi_new]
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_down[i_down] += 1
            end
        end            
    end

    i_down = 1
    psi_down = H.tab_inv_down[i_down]
    psi1_hops_down = differ_by_1_entry(psi_down)
    nn_up = zeros(Int, H.N_up)
    for i_up = 1:H.N_up
        psi_up = H.tab_inv_up[i_up]
        i = (i_down-1)*H.N_up + i_up
        for (psi_new, hop) in psi1_hops_down
            j = (H.tab_down[psi_new]-1)*H.N_up + i_up
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_up[i_up] += 1
            end
        end
        psi1_hops_up = differ_by_1_entry(psi_up)
        for (psi_new, hop) in psi1_hops_up
            j = (i_down-1)*H.N_up + H.tab_up[psi_new]
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                nn_up[i_up] += 1
            end
        end            
    end
    vcat(0, cumsum([ sum(nn_down-(nn_down[1]-n)) for n in nn_up]))
end

function gen_H_upper_step(i_up::Int, H::Hubbard, nn, I, J, x_symm, x_anti)
    n = nn[i_up]
    psi_up = H.tab_inv_up[i_up]
    psi1_hops_up = differ_by_1_entry(psi_up)
    for i_down = 1:H.N_down
        psi_down = H.tab_inv_down[i_down]
        i = (i_up-1)*H.N_down + i_down
        for (psi_new, hop) in psi1_hops_up
            j = (H.tab_up[psi_new]-1)*H.N_down + i_down
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                n += 1
                I[n] = i
                J[n] = j
		s = get_sign_up(psi_new, psi_down, hop)
                x_symm[n] = H.v_symm[hop[1], hop[2]] *s
                x_anti[n] = H.v_anti[hop[1], hop[2]] *s
            end
        end
        psi1_hops_down = differ_by_1_entry(psi_down)
        for (psi_new, hop) in psi1_hops_down
            j = (i_up-1)*H.N_down + H.tab_down[psi_new]
            if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                n += 1
                I[n] = i
                J[n] = j
		s = get_sign_down(psi_up, psi_new, hop)
                x_symm[n] = H.v_symm[hop[1], hop[2]] *s
                x_anti[n] = H.v_anti[hop[1], hop[2]] *s
            end
        end            
    end
end


function gen_H_upper_parallel(H::Hubbard)
    nn = get_dims(H)
    I = SharedArray{Int,1}(H.N_nz)
    J = SharedArray{Int,1}(H.N_nz)
    x_symm = SharedArray{Float64,1}(H.N_nz)
    x_anti = SharedArray{Float64,1}(H.N_nz)
    pmap(n->gen_H_upper_step(n, H, nn, I, J, x_symm, x_anti), 1:H.N_down) 
    nz = 0
    for n=1:nn[end]
        if x_symm[n]!=0.0
            nz += 1
            I[nz] = I[n]
            J[nz] = J[n]
            x_symm[nz] = x_symm[n]
            x_anti[nz] = x_anti[n]
        end
    end
    H.H_upper_symm = sparse(I[1:nz], J[1:nz], x_symm[1:nz], H.N_psi, H.N_psi) 
    H.H_upper_anti = sparse(I[1:nz], J[1:nz], x_anti[1:nz], H.N_psi, H.N_psi) 
end

function gen_H_diag_parallel(H::Hubbard)
    d = SharedArray{Float64,1}(H.N_psi)
    @distributed for i_up = 1:H.N_up 
        psi_up = H.tab_inv_up[i_up]
        x_up = sum([ H.v_symm[k,k] for k=1:H.N_s if psi_up[k] ]) 

        for i_down = 1:H.N_down 
            psi_down = H.tab_inv_down[i_down]
            i = (i_up-1)*H.N_down + i_down
            x = x_up + sum([ H.v_symm[k,k] for k=1:H.N_s if psi_down[k] ]) 
            x += sum([H.U for k=1:H.N_s if psi_down[k] && psi_up[k]])
            d[i] = x
        end
    end
    H.H_diag = sdata(d)
end






function gen_H_upper(H::Hubbard)
    I = zeros(Int, H.N_nz)
    J = zeros(Int, H.N_nz)
    x_symm = zeros(Float64, H.N_nz)
    x_anti = zeros(Float64, H.N_nz)
    n = 0
    for i_up = 1:H.N_up
        psi_up = H.tab_inv_up[i_up]
        psi1_hops_up = differ_by_1_entry(psi_up)
        for i_down = 1:H.N_down
            psi_down = H.tab_inv_down[i_down]
            i = (i_up-1)*H.N_down + i_down
            for (psi_new, hop) in psi1_hops_up
                j = (H.tab_up[psi_new]-1)*H.N_down + i_down
                if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    s = get_sign_up(psi_new, psi_down, hop)
                    x_symm[n] = H.v_symm[hop[1], hop[2]] *s 
                    x_anti[n] = H.v_anti[hop[1], hop[2]] *s 
                end
            end
            psi1_hops_down = differ_by_1_entry(psi_down)
            for (psi_new, hop) in psi1_hops_down
                j = (i_up-1)*H.N_down + H.tab_down[psi_new]
                if j>i && H.v_symm[hop[1], hop[2]]!=0.0
                    n += 1
                    I[n] = i
                    J[n] = j
                    s = get_sign_down(psi_up, psi_new, hop)
                    x_symm[n] = H.v_symm[hop[1], hop[2]] *s
                    x_anti[n] = H.v_anti[hop[1], hop[2]] *s
                end
            end            
        end
    end
    H.H_upper_symm = sparse(I[1:n], J[1:n], x_symm[1:n], H.N_psi, H.N_psi) 
    H.H_upper_anti = sparse(I[1:n], J[1:n], x_anti[1:n], H.N_psi, H.N_psi) 
end

function gen_H_diag(H::Hubbard)
    d = zeros(H.N_psi)
    for i_up = 1:H.N_up 
        psi_up = H.tab_inv_up[i_up]
        x_up = sum([ H.v_symm[k,k] for k=1:H.N_s if psi_up[k] ]) 

        for i_down = 1:H.N_down 
            psi_down = H.tab_inv_down[i_down]
            i = (i_up-1)*H.N_down + i_down
            x = x_up + sum([ H.v_symm[k,k] for k=1:H.N_s if psi_down[k] ]) 
            x += sum([H.U for k=1:H.N_s if psi_down[k] && psi_up[k]])
            d[i] = x
        end
    end
    H.H_diag = d
end


function Hubbard(N_s::Int, n_up::Int, n_down::Int, 
                 v_symm::Array{Float64,2}, v_anti::Array{Float64,2}, 
                 U::Real, f::Function, fd::Function; 
                 store_upper_part_only::Bool=true)
    N_up = binomial(N_s, n_up)
    N_down = binomial(N_s, n_down)
    N_psi = N_up*N_down
    N_nz = div(N_psi*(n_up*(N_s-n_up)+n_down*(N_s-n_down)),2)
    tab_up, tab_inv_up = gen_tabs(N_s, n_up)
    if n_up==n_down
        tab_down = tab_up
        tab_inv_down = tab_inv_up
    else
        tab_down, tab_inv_down = gen_tabs(N_s, n_down)
    end
    H =  Hubbard(N_s, n_up, n_down, N_up, N_down, N_psi, N_nz,
                           v_symm, v_anti, U, Float64[], spzeros(1,1), spzeros(1,1),
                           tab_up, tab_inv_up, tab_down, tab_inv_down,
                           store_upper_part_only, f, fd, 0)
    if nprocs()>1
        gen_H_diag_parallel(H)
        gen_H_upper_parallel(H)
    else
        gen_H_diag(H)
        gen_H_upper(H)
    end
    if !store_upper_part_only
        H.H_upper_symm =  H.H_upper_symm + H.H_upper_symm'
        H.H_upper_anti =  H.H_upper_anti - H.H_upper_anti'
    end
    H
end



function double_occupation(H::Hubbard, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    r = zeros(H.N_s)
    for i_up = 1:H.N_up
        psi_up = H.tab_inv_up[i_up]
        for i_down = 1:H.N_down
            i = (i_up-1)*H.N_down + i_down
            psi_down = H.tab_inv_down[i_down]
            f = abs(psi[i])^2
            for k=1:H.N_s
                if psi_up[k] & psi_down[k]
                    r[k] += f
                end
            end
        end
    end
    r
end

double_occupation(H::HubbardState, psi::Union{Array{Complex{Float64},1},Array{Float64,1}}) = double_occupation(H.H, psi)

"""
Performs matrix multiplication `Y = H * B`.
"""
function LinearAlgebra.mul!(Y, H::HubbardState, B)
    fac_symm = real(H.fac_offdiag)
    fac_anti = imag(H.fac_offdiag)

    if H.H.store_upper_part_only
        if fac_anti == 0.0
            Y[:] = H.fac_diag*(H.H.H_diag.*B) + fac_symm*(H.H.H_upper_symm*B + H.H.H_upper_symm'*B)
        else    
            Y[:] = H.fac_diag*(H.H.H_diag.*B) + fac_symm*(H.H.H_upper_symm*B + H.H.H_upper_symm'*B) +  
                                        (1im*fac_anti)*(H.H.H_upper_anti*B - H.H.H_upper_anti'*B)
        end
    else
        if fac_anti == 0.0
            Y[:] = H.fac_diag*(H.H.H_diag.*B) + fac_symm*(H.H.H_upper_symm*B) 
        else    
            Y[:] = H.fac_diag*(H.H.H_diag.*B) + fac_symm*(H.H.H_upper_symm*B) + (1im*fac_anti)*(H.H.H_upper_anti*B) 
        end
    end
    H.H.counter += 1
end

LinearAlgebra.size(H::Hubbard) = (H.N_psi, H.N_psi)
LinearAlgebra.size(H::Hubbard, dim::Int) = dim<1 ? 
      error("arraysize: dimension out of range") :
      (dim<3 ? H.N_psi : 1)
LinearAlgebra.size(H::HubbardState) = size(H.H)
LinearAlgebra.size(H::HubbardState, dim::Int) = size(H.H, dim)

LinearAlgebra.eltype(H::HubbardState) = imag(H.fac_offdiag)==0.0 ? Float64 : Complex{Float64}
LinearAlgebra.issymmetric(H::HubbardState) = imag(H.fac_offdiag)==0.0 
LinearAlgebra.ishermitian(H::HubbardState) = true
LinearAlgebra.checksquare(H::HubbardState) = H.H.N_psi 


function groundstate(H::HubbardState)
    lambda,g = eigs(H, nev=1, which=:SR)
    lambda = lambda[1]
    real(lambda[1]), g[:,1]
end

function energy(H::HubbardState, psi::Union{Array{Complex{Float64},1},Array{Float64,1}})
    T = imag(H.fac_offdiag)!=0.0 ? Complex{Float64} : eltype(psi)
    psi1 = zeros(T, H.H.N_psi)
    mul!(psi1, H, psi)
    real(dot(psi,psi1))
end

function LinearAlgebra.norm(H::HubbardState, p::Real=2)
    if p==2
        throw(ArgumentError("2-norm not implemented for Hubbard. Try norm(H, p) where p=1 or Inf."))
    elseif !(p==1 || p==Inf)
        throw(ArgumentError("invalid p-norm p=$p. Valid: 1, Inf"))
    end
    # TODO consider also H_upper_sym
    s = zeros(H.H.N_psi)
    for j = 1:H.H.N_psi
        for i = H.H.H_upper_symm.colptr[j]:H.H.H_upper_symm.colptr[j+1]-1
            s[j] += abs(H.H.H_upper_symm.nzval[i])
        end
    end
    if H.H.store_upper_part_only
        for i=1:length(H.H.H_upper_symm.nzval)
            s[H.H.H_upper_symm.rowval[i]] += abs(H.H.H_upper_symm.nzval[i])
        end    
    end
    s[:] *= abs(H.fac_offdiag) 
    s[:] += abs(H.fac_diag)*abs.(H.H.H_diag)
    maximum(s)
end


"""Construct dense matrix for state"""
function full(H::HubbardState)
    fac_symm = real(H.fac_offdiag)
    fac_anti = imag(H.fac_offdiag)

    if H.H.store_upper_part_only
        return (
            diagm(0 => H.fac_diag*(H.H.H_diag)) + fac_symm*(H.H.H_upper_symm + H.H.H_upper_symm') +  
            (1im*fac_anti)*(H.H.H_upper_anti - H.H.H_upper_anti'))
    else
        return (
            diagm(0 => H.fac_diag*(H.H.H_diag)) + fac_symm*(H.H.H_upper_symm) + (1im*fac_anti)*(H.H.H_upper_anti)) 
    end
end

