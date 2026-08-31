# Gated Linear Unit
# 2017 Dauphin, et al
# GLU(γ) = (W1 × γ + b1) ⊗ σ(W2 × γ + b2)

# same as torch.nn.functional.glu
function glu(x::AbstractVector)
    n = length(x)
    @assert n % 2 == 0 "glu requires even lengths"
    n2 = div(n,2)
    a  = @view x[1:n2]
    b  = @view x[n2+1:end]
    map((a,b)->a * sigmoid(b), a, b)
end

# TODO: check if this mutation is allowed by autodiff
function glu(x::AbstractMatrix)
    n,m = size(x)
    @assert n % 2 == 0 "glu requires even lengths"
    n2 = div(n,2)
    y  = Matrix{eltype(x)}(undef,n2,m)
    for i in 1:m
        y[:,i] = glu(@view x[:,i])
    end
    return y
end

# TODO: check if this is allowed by autodiff
function glu(x::AbstractArray)
    Nin,Ntime,Nbatch = size(x)
    x_flat = reshape(x, Nin, Ntime * Nbatch)
    out = Float32[]
    for i in 1:Ntime * Nbatch
        x_i = @view x_flat[:,i]
        append!(out, glu(x_i))
    end
    reshape(out, div(Nin,2), Ntime, Nbatch)
end

struct GatedLinearUnit{T <: Union{Chain, Dense}}
    nn::T
end

function GatedLinearUnit(input_size, hidden_size = nothing; dropout = nothing)
    hidden_size = something(hidden_size, input_size)
    fc = Dense(input_size => hidden_size * 2)
    if isnothing(dropout) || iszero(dropout)
        return GatedLinearUnit(fc)
    else
        return GatedLinearUnit(Chain(fc, Dropout(convert(Float32,dropout))))
    end
end

# input: x (Nin, Ntime, Nbatch)
function (u::GatedLinearUnit{T})(x) where {T}
    glu(u.nn(x))
end
