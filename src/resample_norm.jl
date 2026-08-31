# used in GatedResidualNetwork
struct ResampleNorm
    Nin::Int
    Nout::Int
    resample::TimeDistributedInterpolation
    mask::Vector{Float32}
    norm::LayerNorm
end

Flux.@layer ResampleNorm trainable=(mask,norm)

ResampleNorm(Nin,Nout) = ResampleNorm(
    Nin, Nout, TimeDistributedInterpolation(Nout), zeros(Nout), LayerNorm(Nout))

# x: (Nin, Ntime)  or  (Nin, Ntime, Nbatch)
# resamples Nin to match Nout, then masks and applies LayerNorm
function (u::ResampleNorm)(x)
    println("ResampleNorm Nin:$(u.Nin) Nout:$(u.Nout) x:$(size(x))")
    @assert size(x,1) == u.Nin
    if u.Nin != u.Nout
        if ndims(x)<=2
            x = u.resample(x)
        else
            @assert ndims(x)==3
            Nin, Ntime, Nbatch = size(x)
            x_flat = reshape(x, Nin, Ntime*Nbatch)
            x = u.resample(x_flat)
            x = reshape(x, :, Ntime, Nbatch)
        end
        println("ResampleNorm resampled x:$(size(x))")
    end
    # specialize for vector/matrix?
    activated_mask = Flux.sigmoid_fast.(u.mask) * 2
    println("ResampleNorm activated_mask x:$(size(activated_mask))")
    if ndims(x) <= 2
        x = x .* activated_mask
    elseif ndims(x) == 3
        Nout, Ntime, Nbatch = size(x)
        @assert Nout==u.Nout
        x_flat = reshape(x, u.Nout, :)
        println("ResampleNorm x_flat x:$(size(x_flat))")
        x_flat_masked = x_flat .* activated_mask
        println("ResampleNorm x_flat_masked x:$(size(x_flat_masked))")
        x = reshape(x_flat_masked, u.Nout, Ntime, Nbatch)
    else
        error("unsupported dims:$(ndims(x))")
    end
    return u.norm(x)
end