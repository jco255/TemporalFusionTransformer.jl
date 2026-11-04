module TestGRN

using Test, PythonCall, TFT

np = pyimport("numpy")
pt = pyimport("torch")
pf = pyimport("pytorch_forecasting")

Nin      = 4
Nout     = 3
Nhidden  = 5
Nbatch   = 2
Ncontext = 7

DROPOUT  = 0.0
TOL      = 1f-6

function copyto_check!(a,b)
    @test size(a)==size(b)
    copyto!(a,b)
end

ptov = v -> pyconvert(Vector, v.detach().numpy())
ptom = v -> pyconvert(Matrix, v.detach().numpy())

x_p = pt.rand(Nbatch, Nin)
c_p = pt.rand(Nbatch, Ncontext)
x_j = pyconvert(Matrix, x_p.numpy().transpose())
c_j = pyconvert(Matrix, c_p.numpy().transpose())

# without context
grn_1p = pf.models.temporal_fusion_transformer.sub_modules.GatedResidualNetwork(Nin, Nhidden, Nout, DROPOUT)
grn_1j = TFT.GatedResidualNetwork(Nin, Nhidden, Nout, dropout = DROPOUT)

copyto_check!(grn_1j.fc1.weight, ptom(grn_1p.fc1.weight))
copyto_check!(grn_1j.fc2.weight, ptom(grn_1p.fc2.weight))
copyto_check!(grn_1j.gate_norm.glu.nn.weight, ptom(grn_1p.gate_norm.glu.fc.weight))

# static params
@test pyconvert(Int, grn_1p.input_size ) == grn_1j.Nin
@test pyconvert(Int, grn_1p.hidden_size) == grn_1j.Nhidden
@test pyconvert(Int, grn_1p.output_size) == grn_1j.Nout
@test isnothing(pyconvert(Union{Nothing,Int}, grn_1p.context_size))
@test isnothing(grn_1j.Ncontext)
@test isnothing(grn_1j.gate_norm.add_norm.resample)
@test pyconvert(Float64, grn_1p.dropout) == grn_1j.dropout
@test pyconvert(Bool, grn_1p.residual  ) == grn_1j.residual

# weights
@test grn_1j.gate_norm.glu.nn.bias == ptov(grn_1p.gate_norm.glu.fc.bias)
@test grn_1j.fc1.bias == ptov(grn_1p.fc1.bias)
@test grn_1j.fc2.bias == ptov(grn_1p.fc2.bias)
@test grn_1j.resample_norm.mask            == ptov(grn_1p.resample_norm.mask)
@test grn_1j.resample_norm.norm.diag.bias  == ptov(grn_1p.resample_norm.norm.bias)
@test grn_1j.resample_norm.norm.diag.scale == ptov(grn_1p.resample_norm.norm.weight)
@test ptov(grn_1p.gate_norm.add_norm.norm.weight) == grn_1j.gate_norm.add_norm.norm.diag.scale
@test ptov(grn_1p.gate_norm.add_norm.norm.bias)   == grn_1j.gate_norm.add_norm.norm.diag.bias

# output
grn_1p.forward(x_p)
grn_1j(x_j)
@test (ptom(grn_1p.forward(x_p)) - grn_1j(x_j)' .|> abs |> maximum) < TOL

# with context
grn_2p = pf.models.temporal_fusion_transformer.sub_modules.GatedResidualNetwork(Nin, Nhidden, Nout, DROPOUT, Ncontext)
grn_2j = TFT.GatedResidualNetwork(Nin, Nhidden, Nout, dropout = DROPOUT, Ncontext = Ncontext)

copyto_check!(grn_2j.fc1.weight, ptom(grn_2p.fc1.weight))
copyto_check!(grn_2j.fc2.weight, ptom(grn_2p.fc2.weight))
copyto_check!(grn_2j.context.weight, ptom(grn_2p.context.weight))
copyto_check!(grn_2j.gate_norm.glu.nn.weight, ptom(grn_2p.gate_norm.glu.fc.weight))

@test isnothing(grn_2j.gate_norm.add_norm.resample)

# output 
@test (ptom(grn_2p.forward(x_p, c_p)) - grn_2j(x_j, c_j)' .|> abs |> maximum) < TOL

end # module