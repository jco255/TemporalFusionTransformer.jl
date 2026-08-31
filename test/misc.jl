module TestMisc

using Test, PythonCall, TFT

pt = pyimport("torch")
pf = pyimport("pytorch_forecasting")

TOL = 1f-6

Nout = 3
Nbatch  = 2
tdi_p = pf.models.temporal_fusion_transformer.sub_modules.TimeDistributedInterpolation(Nout, batch_first=true, trainable=false)

x_p = pt.rand(Nbatch,4)
x_j = pyconvert(Matrix,x_p.detach().numpy())
tdi_p.forward(x_p)

tdi_j = TFT.TimeDistributedInterpolation(Nout)
tdi_j(x_j)

y_p = tdi_p.forward(x_p)
y_j = tdi_j(x_j)

@test maximum(abs.(y_j - pyconvert(Matrix, y_p))) < 1e-6

end # module