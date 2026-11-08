## Abstract
This is a Julia version of [PytorchForecasting TemporalFusionTransformer](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/api/pytorch_forecasting.models.temporal_fusion_transformer._tft.TemporalFusionTransformer.html). 
It is everything you want in a forecasting model -- interpretability, static/categorical/known/unknown covariates, variable relevance detection,
nonlinear transformations, encoder/decoder RNN backbone, etc.
Some [Oxford/Google people](https://arxiv.org/pdf/1912.09363) came up with everything.

### Example

```
tft = TFT.TemporalFusionTransformer(;
    hidden_size = 8,
    output_size = 1,
    attention_head_size = 2,
    x_reals = ["x1","x2"],
    time_varying_reals_encoder = ["x1","x2"],
    time_varying_reals_decoder = ["x1",],
)

Ncont = length(tft.x_reals)

x1 = Dict(
    "encoder_cont"    => rand(Float32, Ncont, Nencoder, Nbatch),
    "decoder_cont"    => rand(Float32, Ncont, Ndecoder, Nbatch),
    "encoder_cat"     => Array{Int}(undef, 0, Nencoder, Nbatch),
    "decoder_cat"     => Array{Int}(undef, 0, Ndecoder, Nbatch),
    "encoder_lengths" => fill(Nencoder, Nbatch),
    "decoder_lengths" => fill(Ndecoder, Nbatch),
    )

y1 = rand(Float32, Nbatch) 

loss, grads = Flux.withgradient(tft) do m
    y1_hat = m(x1)
    Flux.mae(y1_hat[1,end,:], y1)
end
```
