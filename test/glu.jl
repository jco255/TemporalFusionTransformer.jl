module TestGLU

using Test, PythonCall, TFT

np = pyimport("numpy")
pt = pyimport("torch")
pf = pyimport("pytorch_forecasting")

Nin = 4
Nhidden = 5
Nbatch  = 2

xj_in = randn(Float32, Nbatch, Nin)
xp_in = np.array(xj_in)
xt_in = pt.Tensor(xp_in)

TOL = 1f-6

@test Bool(np.abs(pt.nn.functional.glu(xt_in.select(0,0)).numpy() - np.array(TFT.glu(xj_in[1,:]))).max() < TOL)
@test Bool(np.abs(pt.nn.functional.glu(xt_in).numpy() - np.array(TFT.glu(xj_in))).max() < TOL)

# python models
glu_p = pf.models.temporal_fusion_transformer.sub_modules.GatedLinearUnit(Nin,Nhidden)
glu_l = pylist(glu_p.parameters())
glu_l[0] # weights Tensor 6 x 3
glu_l[1] # bias    Tensor 6
glu_p_weights = pyconvert(Array{Float32}, glu_l[0].detach().numpy())

# julia models
glu_j = TFT.GatedLinearUnit(Nin,Nhidden)
copyto!(glu_j.nn.weight, glu_p_weights)

y1_p = glu_p.forward(xt_in.select(0,0))
y1_j = glu_j(xj_in[1,:])

y2_p = glu_p.forward(xt_in)
y2_j = glu_j(xj_in')

@test all(abs.(pyconvert(Vector, y1_p.detach().numpy()) - y1_j ) .< TOL )
@test all(abs.(pyconvert(Matrix, y2_p.detach().numpy()) - y2_j') .< TOL )

end # module