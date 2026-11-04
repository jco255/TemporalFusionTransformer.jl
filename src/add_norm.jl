
struct AddNorm 
    resample::Union{Nothing,TimeDistributedInterpolation}
    norm::LayerNorm
end

AddNorm(Nin::Int) = AddNorm(Nin, Nin)
AddNorm(Nin::Int, Nskip::Int) = AddNorm(
    Nin==Nskip ? nothing : TimeDistributedInterpolation(Nin),
    LayerNorm(Nin)
)

# AddNorm layer: adds a skip connection and applies layer normalization
# x: input tensor
# skip: skip connection tensor, aligned if necessary to the dimension of x
function (u::AddNorm)(x, skip)
    Nin = size(x, 1)
    Nskip = size(skip, 1)
    if Nin != Nskip
        skip = u.resample(skip)
    end
    return u.norm(x + skip)
end

# used in GatedResidualNetwork
struct GateAddNorm
    Nskip::Int
    glu::GatedLinearUnit
    add_norm::AddNorm
end

GateAddNorm(Nin, Nhidden, Nskip, dropout) = (
    GateAddNorm(Nskip,
        GatedLinearUnit(Nin, Nhidden; dropout = dropout),
        AddNorm(Nhidden, Nskip)))

function (u::GateAddNorm)(xp, xs)
    u.add_norm(u.glu(xp), xs)
end