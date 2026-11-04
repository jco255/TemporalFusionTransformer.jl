module TestResampleNorm

using Test, PythonCall, Flux, TFT

include("util.jl")

np = pyimport("numpy")
pt = pyimport("torch")
pf = pyimport("pytorch_forecasting")

Nin    = 4
Nout   = 3
Ntime  = 5
Nbatch = 2

const TOL = 1e-5

begin
    rn_p = pf.models.temporal_fusion_transformer.sub_modules.ResampleNorm(Nin, Nout)
    rn_j = TFT.ResampleNorm(Nin, Nout)

    # randomize weights
    rn_j.mask .= rand(Float32, Nout)
    rn_p.mask = pt.nn.parameter.Parameter(pt.Tensor(np.array(rn_j.mask)))

    ## 2D
    x_p = pt.rand(Ntime, Nin)
    x_j = pyconvert(Matrix,x_p.numpy())'

    y_p = rn_p.forward(x_p)
    y_j = rn_j(x_j)

    @test abs.(y_j' - pyconvert(Matrix, y_p.detach().numpy())) |> maximum < TOL

    ## 3D
    x_p = pt.rand(Nbatch, Ntime, Nin)
    y_p = rn_p.forward(x_p)

    x_j = transpose3(pyconvert(Array, x_p.numpy()))
    y_j = rn_j(x_j)

    @test ((pyconvert(Array,y_p.detach().numpy()) |> transpose3) - y_j) |> maximum < 1e-6
end

end