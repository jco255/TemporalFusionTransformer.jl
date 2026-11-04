# GRN(a,c) = LayerNorm(a + GLU(η1))
#       η1 = W1 × η2 + b1
#       η2 = ELU(W2 × a + W3 × c + b2)
# 
# when η1 >> 0 the ELU activation acts as identity
#         << 0 the ELU activation becomes constant, making this a linear layer

struct GatedResidualNetwork
    Nin::Int
    Nhidden::Int
    Nout::Int
    dropout::Float32
    Ncontext::Union{Nothing,Int}
    resample_norm::Union{Nothing, ResampleNorm}
    fc1::Dense
    fc2::Dense
    context::Union{Nothing,Dense}
    gate_norm::GateAddNorm
    residual::Bool
end

function GatedResidualNetwork(Nin, Nhidden, Nout; dropout = 0.1, Ncontext = nothing, residual::Bool = false)
    Nresidual = Nin!=Nout && !residual ? Nin : Nout

    resample_norm = Nout == Nresidual ? nothing : ResampleNorm(Nresidual, Nout)
    context = isnothing(Ncontext) ? nothing : Dense(Ncontext, Nhidden, bias=false) # default glorot/xavier uniform

    fc1 = Dense(Nin     => Nhidden, init=Flux.kaiming_normal)
    fc2 = Dense(Nhidden => Nhidden, init=Flux.kaiming_normal)

    @assert isnothing(resample_norm) || all(iszero, resample_norm.norm.diag.bias)
    @assert all(iszero, fc1.bias)
    @assert all(iszero, fc2.bias)

    gate_norm = GateAddNorm(Nhidden, Nout, Nout, dropout)

    return GatedResidualNetwork(Nin, Nhidden, Nout, dropout, Ncontext,
        resample_norm,
        fc1, fc2,
        context, gate_norm,
        residual)
end

# input: x (Nin, Ntime, Nbatch)
# modes:
#   !context !residual   layer_norm(glu(fc2(elu(fc1(x)          ))) + x)
#   !context  residual   layer_norm(glu(fc2(elu(fc1(x)          ))) + residual)
#    context  residual   layer_norm(glu(fc2(elu(fc1(x) + context))) + residual)
function (u::GatedResidualNetwork)(x, context=nothing, residual=nothing)
    if isnothing(residual)
        residual = x
    end

    if u.Nin != u.Nout && !u.residual
        println("Nin:$(u.Nin) Nout:$(u.Nout) context:$(isnothing(context) ? nothing : size(context)) residual:$(isnothing(residual) ? nothing : size(residual))")
        residual = u.resample_norm(residual)
    end

    x = u.fc1(x)

    if !isnothing(context)
        context = u.context(context)
        x = x + context 
    end

    x = Flux.elu.(x)
    x = u.fc2(x) # Nhidden, Ntime, Nbatch
    x = u.gate_norm(x, residual)
    return x
end
