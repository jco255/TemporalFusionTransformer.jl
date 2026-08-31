module TestNNFunctional

using Test, PythonCall, TFT

pt = pyimport("torch")

t  = pt.rand(3,4)
tj = pyconvert(Array{Float32}, t.numpy())
u  = pt.nn.functional.interpolate(t.unsqueeze(1),5,mode="linear", align_corners=true).squeeze(1)
uj = pyconvert(Array{Float32}, u.numpy())
v  = TFT.interpolate(tj, 5)

@test maximum(abs.(vec(uj - v))) <= 1f-7

end # module