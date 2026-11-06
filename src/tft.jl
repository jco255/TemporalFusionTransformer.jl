""" TemporalFusionTransformer
----------
hidden_size : int, default=16
    hidden size of network which is its main hyperparameter.
    Can range from 8 to 512.
dropout : float, default=0.1
    dropout rate
output_size : int default=7
    number of outputs
loss : MultiHorizonMetric, default=QuantileLoss()
    loss function taking prediction and targets
attention_head_size : int, default=4
    number of attention heads (4 is a good default)
static_reals: names of static continuous variables
static_categoricals: names of static categorical variables
time_varying_categoricals_encoder: names of categorical variables for encoder
time_varying_categoricals_decoder: names of categorical variables for decoder
time_varying_reals_encoder: names of continuous variables for encoder
time_varying_reals_decoder: names of continuous variables for decoder
x_reals: order of continuous variables in tensor passed to forward function
x_categoricals: order of categorical variables in tensor passed to forward function
hidden_continuous_size: default for hidden size for processing continuous variables (similar to categorical
    embedding size)
embedding_sizes: dictionary mapping (string) indices to tuple of number of categorical classes and
    embedding size
embedding_paddings: list of indices for embeddings which transform the zero's embedding to a zero vector
embedding_labels: dictionary mapping (string) indices to list of categorical labels
learning_rate: learning rate
""" 
struct TemporalFusionTransformer5
    hidden_size::Int
    static_reals::Vector{String}
    static_categoricals::Vector{String}
    time_varying_categoricals_encoder::Vector{String}
    time_varying_categoricals_decoder::Vector{String}
    time_varying_reals_encoder::Vector{String}
    time_varying_reals_decoder::Vector{String}
    x_reals::Vector{String}
    x_categoricals::Vector{String}

    static_variable_selection::VariableSelectionNetwork
    encoder_variable_selection::VariableSelectionNetwork
    decoder_variable_selection::VariableSelectionNetwork
    static_context_variable_selection::GatedResidualNetwork
    static_context_initial_hidden_lstm::GatedResidualNetwork
    static_context_initial_cell_lstm::GatedResidualNetwork
    static_context_enrichment::GatedResidualNetwork
    lstm_encoder::LSTM
    lstm_decoder::LSTM
    post_lstm_gate_encoder::GatedLinearUnit
    post_lstm_gate_decoder::GatedLinearUnit
    post_lstm_add_norm_encoder::AddNorm
    post_lstm_add_norm_decoder::AddNorm
    static_enrichment::GatedResidualNetwork
    multihead_attn::MultiHeadAttention
    post_attn_gate_norm::GateAddNorm
    pos_wise_ff::GatedResidualNetwork
    pre_output_gate_norm::GateAddNorm
    output_layer::Dense
end

function TemporalFusionTransformer(;
        hidden_size::Int = 16,
        dropout::Float32 = 0.1f0,
        output_size::Int = 7,
        loss::Function = Flux.Losses.mae,
        attention_head_size::Int = 4,
        static_reals::Vector{String} = Vector{String}(),
        static_categoricals::Vector{String} = Vector{String}(),
        time_varying_categoricals_encoder::Vector{String} = Vector{String}(),
        time_varying_categoricals_decoder::Vector{String} = Vector{String}(),
        time_varying_reals_encoder::Vector{String} = Vector{String}(),
        time_varying_reals_decoder::Vector{String} = Vector{String}(),
        x_reals::Vector{String} = Vector{String}(),
        x_categoricals::Vector{String} = Vector{String}(),
        hidden_continuous_size::Int = 8,
        embedding_sizes::Dict{String, Pair{Int, Int}} = Dict{String, Pair{Int, Int}}(),
        embedding_paddings::Vector{String} = Vector{String}(),
        embedding_labels::Dict{String, Array} = Dict{String, Array}(),
        learning_rate::Float32 = 1f-3,
    )

    # continuous variable processing
    prescalers = Dict{String, Dense}(name => Dense(1, hidden_continuous_size) for name in x_reals)

    # variable selection: static variables
    static_input_sizes = Dict(name => embedding_sizes[name][2] for name in static_categoricals)
    for name in static_reals
        static_input_sizes[name] = hidden_continuous_size
    end

    println("TFT making static_variable_selection with Nhidden:$hidden_size")
    static_variable_selection = VariableSelectionNetwork(
            static_input_sizes, hidden_size;
            input_embedding_flags = Dict(k => true for k in static_categoricals),
            dropout = dropout,
            prescalers = prescalers,
        )

    # variable selection: encoder and decoder
    encoder_input_sizes = Dict{String, Int}(name => embeddings_sizes[name] for name in time_varying_categoricals_encoder)
    for name in time_varying_reals_encoder
        encoder_input_sizes[name] = hidden_continuous_size
    end

    decoder_input_sizes = Dict{String, Int}(name => embeddings_sizes[name] for name in time_varying_categoricals_decoder)
    for name in time_varying_reals_decoder
        decoder_input_sizes[name] = hidden_continuous_size
    end

    encoder_variable_selection = VariableSelectionNetwork(
        encoder_input_sizes, hidden_size;
        input_embedding_flags = Dict(k => true for k in time_varying_categoricals_encoder),
        dropout = dropout,
        context_size = hidden_size,
        prescalers = prescalers,
    )

    decoder_variable_selection = VariableSelectionNetwork(
        decoder_input_sizes, hidden_size;
        input_embedding_flags = Dict(k => true for k in time_varying_categoricals_decoder),
        dropout = dropout,
        context_size = hidden_size,
        prescalers = prescalers,
    )

    # static encoders
    static_context_variable_selection = GatedResidualNetwork(hidden_size, hidden_size, hidden_size; dropout = dropout)
    static_context_initial_hidden_lstm = GatedResidualNetwork(hidden_size, hidden_size, hidden_size; dropout = dropout)
    static_context_initial_cell_lstm = GatedResidualNetwork(hidden_size, hidden_size, hidden_size; dropout = dropout)
    static_context_enrichment = GatedResidualNetwork(hidden_size, hidden_size, hidden_size; dropout = dropout)

    # lstm encoder (history) and decoder (future) for local processing
    lstm_encoder = LSTM(hidden_size=>hidden_size, return_state=true)
    lstm_decoder = LSTM(hidden_size=>hidden_size)

    # skip connection for lstm
    post_lstm_gate_encoder = GatedLinearUnit(hidden_size; dropout = dropout)
    post_lstm_gate_decoder = post_lstm_gate_encoder
    post_lstm_add_norm_encoder = AddNorm(hidden_size)
    post_lstm_add_norm_decoder = post_lstm_add_norm_encoder

    # static enrichment and processing past LSTM
    static_enrichment = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout = dropout, Ncontext = hidden_size)

    # attention for long-range processing (TODO: make Interpretable)
    multihead_attn = MultiHeadAttention(hidden_size, nheads = attention_head_size, dropout_prob = dropout)
    post_attn_gate_norm = GateAddNorm(hidden_size, hidden_size, hidden_size, dropout)
    pos_wise_ff = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout = dropout)

    # output processing -> no dropout at this late stage
    pre_output_gate_norm = GateAddNorm(hidden_size, hidden_size, hidden_size, 0f0)

    output_layer = Dense(hidden_size, output_size)

    return TemporalFusionTransformer5(
        hidden_size,
        static_reals,
        static_categoricals,
        time_varying_categoricals_encoder,
        time_varying_categoricals_decoder,
        time_varying_reals_encoder,
        time_varying_reals_decoder,
        x_reals,
        x_categoricals,
        
        static_variable_selection::VariableSelectionNetwork,
        encoder_variable_selection::VariableSelectionNetwork,
        decoder_variable_selection::VariableSelectionNetwork,
        static_context_variable_selection::GatedResidualNetwork,
        static_context_initial_hidden_lstm::GatedResidualNetwork,
        static_context_initial_cell_lstm::GatedResidualNetwork,
        static_context_enrichment::GatedResidualNetwork,
        lstm_encoder::LSTM,
        lstm_decoder::LSTM,
        post_lstm_gate_encoder::GatedLinearUnit,
        post_lstm_gate_decoder::GatedLinearUnit,
        post_lstm_add_norm_encoder::AddNorm,
        post_lstm_add_norm_decoder::AddNorm,
        static_enrichment::GatedResidualNetwork,
        multihead_attn::MultiHeadAttention,
        post_attn_gate_norm::GateAddNorm,
        pos_wise_ff::GatedResidualNetwork,
        pre_output_gate_norm::GateAddNorm,
        output_layer::Dense,
    )
end


"""
input dimensions: variables x time x n_samples
"""

macro showvar(var)
    svar = string(var)
    return quote
        println("name is $($svar) size is ", size($var))
    end
end

# delme
function transpose3(x)
    @assert ndims(x)==3
    l,m,n = size(x)
    [x[i,j,k] for k in 1:n, j in 1:m, i in 1:l]
end

function create_mask(size, lengths)
    (1:size)' .> lengths
end

# Returns causal mask to apply for self-attention layer.
# Note: currently only causal attention is supported, but this can be extended to allow for non-causal attention.
#   There is value in attending to future forecasts if they are made with knowledge currently available
#   one possibility is here to use a second attention layer for future attention (assuming different effects
#   matter in the future than the past) or alternatively using the same layer but allowing forward attention -
#   i.e. only masking out non-available data and self
function get_attention_mask(encoder_lengths, decoder_lengths)
    @assert length(encoder_lengths) == length(decoder_lengths)
    Nbatch = length(encoder_lengths)
    decoder_length = maximum(decoder_lengths)

    # indices to which is attended
    attend_step = collect(1:decoder_length)
    # indices for which is predicted
    predict_step = collect(1:decoder_length)
    # do not attend to steps to self or after prediction

    decoder_mask = attend_step' .>= predict_step
    decoder_mask = reshape(decoder_mask, 1, decoder_length, decoder_length)
    decoder_mask = repeat(decoder_mask, Nbatch, 1, 1)

    encoder_mask = create_mask(maximum(encoder_lengths), encoder_lengths)
    encoder_mask = reshape(encoder_mask, size(encoder_mask, 1), 1, size(encoder_mask, 2))
    encoder_mask = repeat(encoder_mask, 1, decoder_length, 1)

    cat(encoder_mask, decoder_mask, dims=3)
end

function (self::TemporalFusionTransformer5)(x::Dict{String, Array})
    encoder_lengths = x["encoder_lengths"]::Vector{Int}
    decoder_lengths = x["decoder_lengths"]::Vector{Int}
    # concatenate in time dimension
    x_cat  = cat(x["encoder_cat" ]::Array{Int    , 3}, x["decoder_cat"] ::Array{Int    , 3}, dims=2)
    x_cont = cat(x["encoder_cont"]::Array{Float32, 3}, x["decoder_cont"]::Array{Float32, 3}, dims=2)

    timesteps = size(x_cont, 2)  # encode + decode length
    max_encoder_length = maximum(encoder_lengths) # 7
    println("TFT forward max_encoder_length:$max_encoder_length")

    input_vectors = Dict{String, Array{Float32,3}}() # Nin x Ntime x Nbatch
    # TODO self.input_embeddings(x_cat)
    for (idx,name) in enumerate(self.x_reals)
        input_vectors[name] = x_cont[[idx],:,:]
    end

    # Embedding and variable selection
    static_variables = vcat(self.static_reals, self.static_categoricals)
    if length(static_variables) > 0
        # static embeddings will be constant over entire batch
        static_embedding = Dict(name => input_vectors[name][:, 1, :] for name in static_variables) # Nin x Nbatch
        a = size(static_embedding)
        static_embedding, static_sparse_weights = self.static_variable_selection(static_embedding)
        println("TFT static_embedding from $a to ", size(static_embedding)) # (8, 2)
        println("TFT static_sparse_weights:", size(static_sparse_weights))
    else
        static_embedding = zeros(Float32, self.hidden_size, size(x_cont, 3),)
        static_sparse_weights = zeros(Float32, size(x_cont, 3), 0)
        println("TFT zeroing static, embedding:", size(static_embedding))
    end

    context = self.static_context_variable_selection(static_embedding) #(8, 2)  aka  (Nhidden, Nbatch)
    static_context_variable_selection = reshape(context, self.hidden_size, 1, size(x_cont, 3)) # (8, 1, 2)  aka  (Nhidden, Ntime=1, Nbatch)
    println("TFT forward static_context_variable_selection:", size(static_context_variable_selection)) # 8, 1, 2

    encoder_variables = vcat(self.time_varying_categoricals_encoder, self.time_varying_reals_encoder)
    decoder_variables = vcat(self.time_varying_categoricals_decoder, self.time_varying_reals_decoder)

    embeddings_varying_encoder = Dict(name => input_vectors[name][:,1:max_encoder_length,:] for name in encoder_variables)
    println("TFT forward embeddings_varying_encoder1:", size(embeddings_varying_encoder[encoder_variables[1]])) # 1, 7, 2
    embeddings_varying_encoder, encoder_sparse_weights =
        self.encoder_variable_selection(embeddings_varying_encoder, static_context_variable_selection)

    embeddings_varying_decoder = Dict(name => input_vectors[name][:,max_encoder_length+1:end,:] for name in decoder_variables)
    embeddings_varying_decoder, decoder_sparse_weights =
        self.decoder_variable_selection(embeddings_varying_decoder, static_context_variable_selection)

    println("TFT forward context:", size(context)) # 8, 2
    println("TFT forward static_embedding:", size(static_embedding)) # 8, 2
    println("TFT forward embeddings_varying_encoder:", size(embeddings_varying_encoder)) # 8, 7, 2
    println("TFT forward embeddings_varying_decoder:", size(embeddings_varying_decoder)) # 8, 1, 2

    # LSTM
    # inputs: x (Nin, Nlen) or (Nin, Nlen, Nbatch)
    #         h (Nhidden) or (Nhidden, Nbatch)

    # calculate initial state using GRNs
    input_hidden = self.static_context_initial_hidden_lstm(static_embedding)
    input_cell   = self.static_context_initial_cell_lstm(static_embedding)

    println("TFT forward input_hidden:", size(input_hidden)) # 8, 2
    println("TFT forward input_cell:"  , size(input_cell)  ) # 8, 2

    # run local encoder/decoder
    encoder_output, (hidden, cell) = self.lstm_encoder(embeddings_varying_encoder, (input_hidden, input_cell)) # (8,7,2), ((8,2), (8,2))
    decoder_output = self.lstm_decoder(embeddings_varying_decoder, (hidden, cell))
    println("TFT forward encoder_output:", size(encoder_output)) # 8, 7, 2
    println("TFT forward decoder_output:", size(decoder_output)) # 8, 3, 2

    # skip connection over lstm
    lstm_output_encoder = self.post_lstm_gate_encoder(encoder_output)
    lstm_output_encoder = self.post_lstm_add_norm_encoder(lstm_output_encoder, embeddings_varying_encoder)
    println("TFT forward lstm_output_encoder:", size(lstm_output_encoder)) # 8, 7, 2

    lstm_output_decoder = self.post_lstm_gate_decoder(decoder_output)
    lstm_output_decoder = self.post_lstm_add_norm_decoder(lstm_output_decoder, embeddings_varying_decoder)
    println("TFT forward lstm_output_decoder:", size(lstm_output_decoder)) # 8, 3, 2

    lstm_output = cat(lstm_output_encoder, lstm_output_decoder, dims=2)
    println("TFT forward lstm_output:", size(lstm_output)) # 8, 10, 2

    # static enrichment
    static_context_enrichment = self.static_context_enrichment(static_embedding) # GRN
    static_context_enrichment = reshape(static_context_enrichment, self.hidden_size, 1, size(static_context_enrichment, 2))
    attn_input = self.static_enrichment(lstm_output, static_context_enrichment) # GRN
    println("TFT forward static_context_enrichment:", size(static_context_enrichment))
    println("TFT forward attn_input:", size(attn_input)) # 8, 10, 2

    # attention 
    # TODO: make interpretable, i.e. share value weights across each head
    attention_mask = get_attention_mask(encoder_lengths, decoder_lengths)
    println("TFT forward attention_mask:", size(attention_mask)) # (2, 3, 10)  aka  Nbatch, Ndecoder, Ntime
    attn_output, attn_output_weights = self.multihead_attn(
        attn_input[:, max_encoder_length+1:end, :],  # query only for predictions
        attn_input, # key
        attn_input, # value
        mask = transpose3(attention_mask)
    )
    println("TFT fwd attn_output1:", size(attn_output))
    println("TFT fwd attn_output_weights:", size(attn_output_weights))

    # skip connection over attention
    attn_output = self.post_attn_gate_norm(attn_output, attn_input[:, max_encoder_length+1:end,:])
    println("TFT fwd attn_output2:", size(attn_output))

    output = self.pos_wise_ff(attn_output)
    println("TFT fwd output1:", size(output))

    # skip connection over temporal fusion decoder (not LSTM decoder
    # despite the LSTM output contains
    # a skip from the variable selection network)
    output = self.pre_output_gate_norm(output, lstm_output[:, max_encoder_length+1:end, :])
    println("TFT fwd output2:", size(output))

    output = self.output_layer(output)
    println("TFT fwd output3:", size(output))

    """
    return self.to_network_output(
        prediction=self.transform_output(output, target_scale=x["target_scale"]),
        encoder_attention=attn_output_weights[..., :max_encoder_length],
        decoder_attention=attn_output_weights[..., max_encoder_length:],
        static_variables=static_sparse_weights,
        encoder_variables=encoder_sparse_weights,
        decoder_variables=decoder_sparse_weights,
        decoder_lengths=decoder_lengths,
        encoder_lengths=encoder_lengths,
    )
    """

    return output
end