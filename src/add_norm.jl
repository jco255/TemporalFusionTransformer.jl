
struct AddNorm 
    resample::Union{Nothing,TimeDistributedInterpolation}
    norm::LayerNorm
end

AddNorm(Nin::Int) = AddNorm(Nin, Nin)
AddNorm(Nin::Int, Nskip::Int) = AddNorm(
    Nin==Nskip ? nothing : TimeDistributedInterpolation(Nin),
    LayerNorm(Nin)
)

function (u::AddNorm)(x, skip) # torch convention
    # @assert ndims(x)==2 "x:$(size(x))"
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
    u.add_norm(u.glu(xp),xs)
end