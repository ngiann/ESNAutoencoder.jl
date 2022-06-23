
function ae(Y, labels; esn = esn, M = 10, H = 2, iterations = 1, α = 1e-2)

    D = esn.N + 1

    net = [layer(D, M, identity); layer(M, H); layer(H, M, identity); layer(M, D)]

    # initial weights
    ae(Y, labels, randn(numweights(net))*0.1; esn = esn, M = M, H = H, iterations = iterations, α = α)

end

function ae(Y, labels, weights; esn = esn, M = 10, H = 2, iterations = 1, depth = 1, α = 1e-2)

    inputs, targets = [y[1:end-1] for y in Y], [y[2:end] for y in Y]

    #-------------------------------------------------
    # Sort out constants
    #-------------------------------------------------

    D = esn.N + 1

    N = length(Y); @assert(length(labels) == N)

    @printf("There are %d data items\n", N)


    #-------------------------------------------------
    # collect hidden states
    #-------------------------------------------------

    X = [esn(y) for y in inputs]

    W = [DeterministicESN.getreadouts(esn, y,  1e-2, 0) for y in inputs]


    #-------------------------------------------------
    # setup network
    #-------------------------------------------------

    net = [layer(D, M, identity); layer(M, H); layer(H, M, identity); layer(M, D)]

    @printf("There are %d weights\n", numweights(net))


    #-------------------------------------------------
    # define objective and AD gradient
    #-------------------------------------------------


    function objective(weights)

        local F = α*sum(abs2.(weights))

        @inbounds for n in 1:N

            local W̃ = net(weights, W[n])

            F += sum((targets[n]- vec(X[n] * W̃)).^2)

        end

        return F

    end


    # pre-record a GradientTape
    compiled_f_tape = ReverseDiff.compile(ReverseDiff.GradientTape(objective, vec(weights)))

    function g!(results, inputs)
        ReverseDiff.gradient!(results, compiled_f_tape, inputs)
    end



    #-------------------------------------------------
    # Optimisation
    #-------------------------------------------------

    # options for optimiser
    opt = Optim.Options(iterations = iterations, show_trace = true, show_every=50)

    # call optimiser
    result  = Optim.optimize(objective, g!, vec(weights), LBFGS(), opt)

    # set optimised weight parameters
    setparam!(net, result.minimizer)

    Z = reduce(hcat, [net[3:end](w) for w in W])

    # @show size(Z)
    # # plot
    # figure(0); cla()
    # for l in unique(labels)
    #     idx = findall(labels .== l)
    #     plot(Z[1, idx], Z[2, idx], "o")
    # end

    return Z, result.minimizer
end
