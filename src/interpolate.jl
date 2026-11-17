function interpolate(yin::AbstractVector, Nout)
    Nin = length(yin)
    map(1:Nout) do i
        if i == 1
            return first(yin)
        elseif i == Nout
            return last(yin)
        end
        xin = 1 + (i-1) * (Nin-1)/(Nout-1)
        x0 = floor(Int, xin)
        t = xin - x0
        return yin[x0] + t * (yin[x0 + 1] - yin[x0])
    end
end    

function interpolate(input::AbstractMatrix{T}, Nout) where {T}
    # simplified version of torch.nn.functional.interpolate
    # hardcoded args:
    #     mode = "linear"
    #     align_corners = True
    #     scale_factor = None
    #     antialias = False

    # note: torch.nn.function includes a channels dimension
    # input shape: width x batch
    Nin, Nbatch = size(input)

    out = [interpolate(input[:,i], Nout) for i in 1:Nbatch]
    out = reduce(vcat, out)
    out = reshape(out, Nout, :)
    return out
end

# used in AddNorm and ResampleNorm
struct TimeDistributedInterpolation
    Nin::Int
    Nout::Int
    matrix::Matrix{Float32}
end

Flux.@layer TimeDistributedInterpolation trainable=()

function TimeDistributedInterpolation(Nin,Nout)
    matrix = zeros(Float32,Nin,Nout)
    for i in 1:Nout
        if i == 1
            matrix[1,i] = 1
        elseif i == Nout
            matrix[Nin,i] = 1
        else
            xin = 1 + (i-1) * (Nin-1)/(Nout-1)
            x0 = floor(Int, xin)
            t = xin - x0
            matrix[x0,i] = 1 - t
            matrix[x0 + 1,i] = t
        end
    end
    return TimeDistributedInterpolation(Nin,Nout,matrix')
end

# interpolates along the leading dimension
function (u::TimeDistributedInterpolation)(x)
    if ndims(x) <= 2
        out = u.matrix * x
        println("u:$(u.Nout) $(ndims(x)) size:$(size(x)) out:$(size(out))")
        return out

    elseif ndims(x) == 3
        Nin, Ntime, Nbatch = size(x)
        x_flat = reshape(x, Nin, Ntime * Nbatch)
        y_flat = u.matrix * x_flat
        println("TDI x:$(size(x)) x_flat:$(size(x_flat)) y_flat:$(size(y_flat)) matrix:$(size(u.matrix))")
        out = reshape(y_flat, u.Nout, Ntime, Nbatch)
        println("TDI u:$(u.Nout) $(ndims(x)) size:$(size(x)) out:$(size(out))")
        return out
    end
    error("cant handle shape x:$(size(x))")
end