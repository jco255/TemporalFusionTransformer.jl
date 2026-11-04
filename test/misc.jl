module TestMisc

using Test, PythonCall, TFT

pt = pyimport("torch")
pf = pyimport("pytorch_forecasting")

TOL = 1f-6

Nin    = 4
Nout   = 3
Nbatch = 2

tdi_p = pf.models.temporal_fusion_transformer.sub_modules.TimeDistributedInterpolation(Nout, batch_first=true, trainable=false)
tdi_j = TFT.TimeDistributedInterpolation(Nout)

x_p = pt.rand(Nbatch, Nin)
x_j = pyconvert(Matrix,x_p.detach().numpy())'

y_p = tdi_p.forward(x_p)
y_j = tdi_j(x_j)

@test maximum(abs.(y_j' - pyconvert(Matrix, y_p))) < TOL

end # module