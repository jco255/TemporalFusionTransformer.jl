## Abstract
This is a Julia version of [PytorchForecasting TemporalFusionTransformer](https://pytorch-forecasting.readthedocs.io/en/v1.4.0/api/pytorch_forecasting.models.temporal_fusion_transformer._tft.TemporalFusionTransformer.html). 
It is everything you want in a forecasting model -- interpretability, static/categorical/known/unknown covariates, variable relevance detection,
nonlinear transformations, encoder/decoder RNN backbone, etc.
Some [Oxford/Google people](https://arxiv.org/pdf/1912.09363) came up with everything.

### Example

```
tft = TFT.TemporalFusionTransformer(; hidden_size = 8,
    output_size = 1,
    attention_head_size = 2,
    history_length = 7,
    predict_length = 3,
    time_varying_known_reals = ["x1"],
    time_varying_unknown_reals = ["x2"],
)

n   = 1000
df1 = DataFrame(x1=rand(n), x2=rand(n))
dl  = TFT.DataLoader(df1, tft)
x   = TFT.getobs(dl, [1,2,3])

y1_hat = tft(x)
y1     = rand(Float32, size(x["x_cont"],3)) 
Flux.mae(y1_hat[1,end,:], y1)

loss, grads = Flux.withgradient(tft) do m
    y1_hat = m(x)
    Flux.mae(y1_hat[1,end,:], y1)
end
```
