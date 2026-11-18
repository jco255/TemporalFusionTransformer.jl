
struct DataLoader
    df::DataFrame
    tft::TemporalFusionTransformer7
end

Base.length(dl::DataLoader) = nrow(dl.df) - dl.tft.encoder_length - dl.tft.decoder_length + 1

function Base.getindex(dl::DataLoader, i::Int)
    @view dl.df[i:i + dl.tft.encoder_length + dl.tft.decoder_length - 1, :]
end

import MLUtils: getobs
function MLUtils.getobs(dl::DataLoader, batch_idx::Vector{Int})
    Nbatch = length(batch_idx)
    Nencoder = dl.tft.encoder_length
    Ndecoder = dl.tft.decoder_length
    x_cont = Vector{Float32}(undef,0)
    Nreals = length(dl.tft.x_reals)
    Ncats  = length(dl.tft.x_categoricals)
    for batch_i in batch_idx
        df = dl[batch_i]
        for col in dl.tft.x_reals
            append!(x_cont, df[!,col])
        end
    end
    @assert iszero(Ncats)
    return Dict(
        "x_cont" => reshape(x_cont, Nencoder+Ndecoder, Nreals, Nbatch),
        "x_cat"  => Array{Int}(undef, Nencoder+Ndecoder, 0, Nbatch),
    )
end