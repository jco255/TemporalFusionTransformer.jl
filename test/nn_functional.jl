module TestNNFunctional

using Test, PythonCall, TFT

pt = pyimport("torch")

Nin    = 3
Nbatch = 4
Nout   = 5

x_p = pt.rand(Nbatch, Nin)
x_j = pyconvert(Matrix, x_p.numpy())'
y_p = pt.nn.functional.interpolate(x_p.unsqueeze(1), Nout, mode="linear", align_corners=true).squeeze(1)
y_j = TFT.interpolate(x_j, Nout)

TOL = 1f-7
@test maximum(abs.(pyconvert(Array, y_p.numpy()) - y_j')) <= TOL

end # module