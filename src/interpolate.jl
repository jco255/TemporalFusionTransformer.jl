    
function interpolate(input::AbstractMatrix{T}, Nout) where {T}
    # simplified version of torch.nn.functional.interpolate
    # hardcoded args:
    #     mode = "linear"
    #     align_corners = True
    #     scale_factor = None
    #     antialias = False

    # note: torch.nn.function includes a channels dimension
    # input shape: width x batch
    @assert ndims(input)==2
    Nin, Nbatch = size(input)

    x_in  = range(1,Nin)
    x_out = range(1,Nin, length=Nout)

    y = Matrix{T}(undef, Nout, Nbatch)
    for i in 1:Nbatch
        y_in = @view input[:,i]
        for j in 1:Nout
            y[j,i] = LinearInterpolations.interpolate(x_in, y_in, x_out[j])
        end
    end
    return y
end

# used in AddNorm and ResampleNorm
struct TimeDistributedInterpolation
    Nout::Int
end

# interpolates along the leading dimension
function (u::TimeDistributedInterpolation)(x)
    if ndims(x) <= 2
        return interpolate(x, u.Nout)
    elseif ndims(x) == 3
        Nin, Ntime, Nbatch = size(x)
        x_flat = reshape(x, Nin, Ntime * Nbatch)
        y_flat = interpolate(x_flat, u.Nout)
        return reshape(y_flat, u.Nout, Ntime, Nbatch)
    end
    error("cant handle shape x:$(size(x))")
end