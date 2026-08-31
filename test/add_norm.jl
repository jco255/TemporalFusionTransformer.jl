module TestAddNorm

using Test, PythonCall, Flux, TFT

pt = pyimport("torch")
pf = pyimport("pytorch_forecasting")

Nin     = 4
Nhidden = 5
Nbatch  = 2
Nskip   = 3

const TOL = 1e-6

begin
    xh_p = pt.rand(Nbatch, Nhidden)
    xs_p = pt.rand(Nbatch, Nhidden)
    xh_j = pyconvert(Matrix, xh_p.numpy())
    xs_j = pyconvert(Matrix, xs_p.numpy())

    an_p = pf.models.temporal_fusion_transformer.sub_modules.AddNorm(Nhidden, Nhidden, trainable_add = false)
    an_j = TFT.AddNorm(Nhidden, Nhidden)
    y_p  = an_p.forward(xh_p, xs_p)
    y_j  = an_j(xh_j, xs_j)

    @test all(abs.(pyconvert(Matrix, y_p.detach().numpy()) - y_j) .< TOL)
end

begin
    xh_p = pt.rand(Nbatch, Nhidden)
    xs_p = pt.rand(Nbatch, Nskip)
    xh_j = pyconvert(Matrix, xh_p.numpy())
    xs_j = pyconvert(Matrix, xs_p.numpy())

    an_p = pf.models.temporal_fusion_transformer.sub_modules.AddNorm(Nhidden, Nskip, trainable_add = false)
    an_j = TFT.AddNorm(Nhidden, Nskip)
    y_p  = an_p.forward(xh_p, xs_p)
    y_j  = an_j(xh_j, xs_j)

    @test all(abs.(pyconvert(Matrix, y_p.detach().numpy()) - y_j) .< TOL)
end

begin
    xp_p = pt.rand(Nbatch, Nin)
    xs_p = pt.rand(Nbatch, Nskip)
    xp_j = pyconvert(Array, xp_p.numpy())
    xs_j = pyconvert(Array, xs_p.numpy())

    # python model
    gan_p = pf.models.temporal_fusion_transformer.sub_modules.GateAddNorm(Nin, Nhidden, Nskip)

    # julia model
    gan_j = TFT.GateAddNorm(Nin, Nhidden, Nskip, 0.0)

    # make weights identical
    gan_p_parms = pylist(gan_p.glu.parameters())
    gan_p_weights = pyconvert(Array{Float32}, gan_p_parms[0].detach().numpy())
    copyto!((gan_j.glu.nn::Flux.Dense).weight, gan_p_weights)

    py_result = gan_p.forward(xp_p, xs_p)
    jl_result = gan_j(xp_j', xs_j)

    @test maximum(abs.(pyconvert(Matrix, py_result.detach().numpy()) - jl_result)) < TOL
end

end # module