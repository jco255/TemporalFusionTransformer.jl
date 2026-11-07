
struct VariableSelectionNetwork
    names::Vector{String}
    input_sizes::Dict{String, Int}
    hidden_size::Int
    input_embedding_flags::Dict{String, Bool}
    dropout::Float32
    context_size::Union{Nothing, Int}
    single_variable_grns::Dict{String, TFT.GatedResidualNetwork}
    single_variable_rns::Dict{String, TFT.ResampleNorm}
    prescalers::Dict{String, Dense}

    flattened_grn::Union{Nothing, TFT.GatedResidualNetwork}
end

function VariableSelectionNetwork(
    input_sizes::Dict{String,Int},
    hidden_size::Int;
    input_embedding_flags::Dict{String, Bool} = Dict{String, Bool}(),
    dropout::Float32 = 0.1f0,
    context_size::Union{Nothing,Int} = nothing,
    single_variable_grns_arg::Dict{String, TFT.GatedResidualNetwork} = Dict{String, TFT.GatedResidualNetwork}(),
    prescalers::Dict{String, Dense} = Dict{String, Dense}(),
)
    input_size_total = input_sizes |> values |> sum
    println("input_size_total:$input_size_total")

    flattened_grn = nothing
    if length(input_sizes) > 1
        flattened_grn = TFT.GatedResidualNetwork(
            input_size_total,
            min(hidden_size, length(input_sizes)),
            length(input_sizes);
            dropout = dropout,
            Ncontext = context_size,
            residual = false,
        )
    end
    single_variable_grns = Dict{String, TFT.GatedResidualNetwork}()
    single_variable_rns = Dict{String, TFT.ResampleNorm}()
    prescalers_arg = Dict{String, Dense}()
    for (name, input_size) in input_sizes
        if haskey(single_variable_grns_arg, name)
            single_variable_grns[name] = single_variable_grns_arg[name]
        elseif haskey(input_embedding_flags, name)
            single_variable_rns[name] = TFT.ResampleNorm(input_size, hidden_size)
        else
            single_variable_grns[name] = TFT.GatedResidualNetwork(
                input_size,
                min(input_size, hidden_size),
                hidden_size,
                dropout = dropout,
            )
        end
        if haskey(prescalers, name)  # reals need to be first scaled up
            prescalers_arg[name] = prescalers[name]
        elseif !haskey(input_embedding_flags, name)
            prescalers_arg[name] = Dense(1=>input_size)
        end
    end

    return VariableSelectionNetwork(
        collect(keys(input_sizes)),
        input_sizes,
        hidden_size,
        deepcopy(input_embedding_flags),
        dropout,
        context_size,
        single_variable_grns,
        single_variable_rns,
        prescalers,
        flattened_grn,
    )
end

function (self::VariableSelectionNetwork)(x::Dict, context::Union{Nothing, Array} = nothing)
    # input size is 1 x encoder_len x batch
    get_var = name -> begin
        variable_embedding = x[name]
        if haskey(self.prescalers, name)
            variable_embedding = self.prescalers[name](variable_embedding)
        end
        var_output = self.single_variable_grns[name](variable_embedding)
        variable_embedding => var_output
    end

    if length(self.input_sizes) > 1
        # transform single variables
        var_out_and_weight_in = [get_var(name) for name in self.names]
        weight_inputs = [first(pair) for pair in var_out_and_weight_in]
        var_outputs   = [last(pair)  for pair in var_out_and_weight_in]

        var_outputs = stack(var_outputs, dims=1)
        println("VSN var_outputs:", size(var_outputs)) # (3, 5, 7, 2)    python: (Nbatch, Nencoder, Nhidden, Nvariables)

        # calculate variable weights
        flat_embedding = cat(weight_inputs..., dims=1)
        println("VSN flat_embedding:", size(flat_embedding)) # (3*4, 7, 2)
        sparse_weights = self.flattened_grn(flat_embedding, context)
        a = size(sparse_weights)                                        # (3, 7, 2)
        sparse_weights = reshape(softmax(sparse_weights), a[1], 1, a[2], a[3])
        println("VSN sparse_weights from $a to ", size(sparse_weights)) # (3, 1, 7, 2)  (Nvars, 1, Nencoder, Nbatch)

        outputs = var_outputs .* sparse_weights
        a = size(outputs)                                  # (3, 5, 7, 2)    (Nvars, Nhidden, Nencoder, Nbatch)
        outputs = dropdims(sum(outputs, dims=1), dims=1)
        println("VSN outputs from $a to ", size(outputs))  # (5, 7, 2)   (Nhidden, Nencoder, Nbatch)

    elseif length(self.input_sizes) == 1
        # for one input, do not perform variable selection but just encoding
        name = keys(self.single_variable_grns) |> first
        variable_embedding = x[name]
        if haskey(self.prescalers, name)
            a = size(variable_embedding)
            variable_embedding = self.prescalers[name](variable_embedding)
            b = size(variable_embedding)
            println("VSN transforming $a to $b") # (1, 7, 2) -> (4, 7, 2) aka (Nin, Nencoder, Nbatch)
        end
        outputs = self.single_variable_grns[name](variable_embedding)
        println("VSN outputs: $(size(outputs))") # (5, 7, 2)  aka  (Nhidden, Nencoder, Nbatch)
        # fast forward if only one variable
        if ndims(outputs) == 3  # -> Nvars, Nhidden, Nencoder, Nbatch
            sparse_weights = ones(1, size(outputs, 1), size(outputs, 2), size(outputs, 3)) 
        else  # ndim == 2 -> hidden size, n_variables, Nbatch
            sparse_weights = ones(1, size(outputs, 1), size(outputs, 2))
        end
    else  # for no input
        outputs = zeros(size(context)...)
        if ndims(outputs) == 3  # -> Nvars, Nhidden, Nencoder, Nbatch
            sparse_weights = zeros(0, size(outputs, 1), size(outputs, 2), size(outputs, 3))
        else  # ndims == 2 -> Nvars, Nhidden, Nencoder
            sparse_weights = zeros(0, size(outputs, 1), size(outputs, 2))
        end
    end
    return outputs, sparse_weights
end