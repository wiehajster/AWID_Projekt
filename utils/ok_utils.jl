#Initialization of a layers, activation function, cross-entropy loss and one_hot matrix
mutable struct fullyconnected
    weight
    bias
    act_fn
    fullyconnected(dim_in, dim_out, act_fn)=new(randn(dim_out, dim_in).*0.1, zeros(dim_out, 1), act_fn)
end;

struct ReLU
end;
forward(x::Array{Float64, 2})::Array{Float64,2} = x.*(x.>0)
gradient(x::Array{Float64, 2})::Array{Float64,2} = Array{Float64, 2}(x.>0)

struct Softmax
    #forward method requires inputs as probabilities
    #backward method requires inputs as probabilities
end;
forward(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = softmax(x)
gradient(x::Array{Float64, 2}, act_fn::Softmax)::Array{Float64,2} = (softmax(x)) .* (1 .- softmax(x))

function softmax(x::Array{Float64,2})::Array{Float64,2}
    #converts real numbers to probabilities
    c=maximum(x)
    p = x .- log.( sum( exp.(x .- c) ) ) .-c
    p = exp.(p)
    return p
end

function create_one_hot(data)
    one_hot=zeros(Float64, maximum(data)+1, size(data, 1));
    for i in 1:size(data, 1)
        label=data[i]+1
        one_hot[label, i]=1
    end;
    return one_hot
end;

xe_loss(y_hat::Array{Float64,2}, y::Array{Float64,2}) = -sum(y.*log.(y_hat))
xe_loss_derivative(y_hat::Array{Float64,2}, y::Array{Float64,2}) = y_hat - y

function forward(x::Array{Float64,2}, net::Dict)::Array{Float64,2}
    pre_act=[]
    post_act=[x]
    for n in 1:length(net["Layers"])
        W = net["Layers"][n].weight
        B = net["Layers"][n].bias
        act_fn = net["Layers"][n].act_fn
        value = W*x + B
        x = forward(value)
        append!(pre_act, [value])
        append!(post_act, [x])
    end
    net["Pre-Activation"]=pre_act
    net["Post-Activation"]=post_act
    return x
end; 

function get_grad(dA, W, B, Z, A_prev, act_fn)
    dZ = dA.*gradient(Z)
    dB = dZ
    dW = (dZ * A_prev')
    dA_prev = W'*dZ
    out=[dA_prev, dW, dB]
    return out
end

function accumulate_grad(grad, net, d)
    dA_prev = grad
    dW=[]
    dB=[]
    depth=2
    for n in 1:depth
        
        n_curr = depth-(n-1);
        n_prev = depth-n;
        W = net["Layers"][n_curr].weight;
        B = net["Layers"][n_curr].bias;
        act_fn = net["Layers"][n_curr].act_fn;
        dA = dA_prev;
        pre_act = net["Pre-Activation"][n_curr];
        post_act_prev = net["Post-Activation"][n_curr];

        out = get_grad(dA, W, B, pre_act, post_act_prev, act_fn)     
        
        dA_prev = out[1]
        append!(dW, [out[2]])
        append!(dB, [out[3]]) 
    end

    dW=reverse(dW)
    dB=reverse(dB)

    if d == -1
        d=[dW, dB]
    else
        for n in 1:depth
            @views d[1][n]=d[1][n].+dW[n]
            @views d[2][n]=d[2][n].+dB[n]
        end
    end
    return d

end;

function update(gradients, learning_rate, net)
    for i in 1:length(net["Layers"])
        @views W = net["Layers"][i].weight-=(learning_rate)*gradients[1][i];
        @views B = net["Layers"][i].bias-=(learning_rate)*gradients[2][i];
    end
end;

function train(net, train_x, test_x, train_y_one_hot, test_y_one_hot, mb_size, learning_rate, epochs, train_size)
    
    avg_epoch_train_loss=[]
    avg_epoch_train_acc=[]
    avg_epoch_test_loss=[]
    avg_epoch_test_acc=[]

    for epoch in 1:epochs

        epoch_train_loss=[]
        epoch_test_loss=[]
        epoch_train_correct=0.0
        epoch_test_correct=0.0

        mb_counter=1;
        loss=0.0;

        d=-1
        progress = (epoch-1)/epochs*100
        println("progress = ", progress, "%")

        for i in 1:train_size

            x_tr_resh = reshape(train_x[:, :, i], 784, 1);
            y_hat_tr = forward(x_tr_resh, net);
            y_tr = reshape(train_y_one_hot[:, i], 10, 1);

            # track loss and loss gradient
            loss_grad = xe_loss_derivative(y_hat_tr, y_tr)./mb_size
            d = accumulate_grad(loss_grad, net, d)
            # accumulate loss and accuracy
            loss = loss.+(xe_loss(y_hat_tr, y_tr)./mb_size)
            epoch_train_correct+=(argmax(y_hat_tr) == argmax(y_tr))
            
            #averaging and reseting values if reached end of the minibatch
            if mb_counter%mb_size==0 
                update(d, learning_rate, net)
                append!(epoch_train_loss, loss)
                d=-1;
                mb_counter=0;
                loss=0.0;
            end
            mb_counter+=1
        end

        # get averages for training statistics 
        append!(avg_epoch_train_loss, sum(epoch_train_loss)/length(epoch_train_loss))
        append!(avg_epoch_train_acc, epoch_train_correct/size(train_y,1))

        for i in 1:size(test_y,1)
            # Grab test data point
            x_te=test_x[:,:,i];
            x_te=reshape(x_te,784,1);
            y_hat_te=forward(x_te, net);
            y_te=reshape(test_y_one_hot[:,i], 10, 1);

            # accumulate loss and accuracy
            append!(epoch_test_loss, xe_loss(y_hat_te,y_te))
            epoch_test_correct+=(argmax(y_hat_te)==argmax(y_te))

        end
        
        append!(avg_epoch_test_loss, sum(epoch_test_loss)/length(epoch_test_loss))
        append!(avg_epoch_test_acc, epoch_test_correct/size(test_y,1))

    end

    results=Dict("train_loss"=>avg_epoch_train_loss, 
                 "test_loss"=>avg_epoch_test_loss, 
                 "train_acc"=>avg_epoch_train_acc,
                 "test_acc"=>avg_epoch_test_acc)
    println("progress = 100%")
    return results
    
end

