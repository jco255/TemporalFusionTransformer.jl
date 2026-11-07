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

function glu(x::AbstractMatrix)
    n,m = size(x)
    @assert n % 2 == 0 "glu requires even lengths"
    n2 = div(n,2)
    out = [glu(@view x[:,i]) for i in 1:m]
    out = reduce(vcat, out)
    y = reshape(out, n2, m)

    return y
end

function glu(x::AbstractArray)
    Nin,Ntime,Nbatch = size(x)
    x_flat = reshape(x, Nin, Ntime * Nbatch)
    out = [glu(@view x_flat[:,i]) for i in 1:Ntime * Nbatch ]
    out = reduce(vcat, out)
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
