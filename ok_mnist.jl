include("ok_utils.jl")
using BenchmarkTools
using LinearAlgebra
using MLDatasets

function train(net, train_x, test_x, train_y_one_hot, test_y_one_hot, mb_size, learning_rate, epochs)
    
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

        for i in 1:size(train_y,1)

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

# Requires MLDatasets for training

net=Dict("Layers"=>[], "Pre-Activation"=>[], "Post-Activation"=>[])

#input data pictures with size 28x28 px = 784
#ouput data value from 0 to 9
#hidden layer of 26 neurons

hidden_neurons = 26
mb_size = 30
learning_rate = 0.01
epochs = 1

layers=[]
append!(layers, [fullyconnected(784, hidden_neurons,ReLU())])
append!(layers, [fullyconnected(hidden_neurons, 10, Softmax())])
net["Layers"]=layers;

#Loading data
train_x, train_y = MNIST.traindata(Float64);
test_x, test_y = MNIST.testdata(Float64);
#Creating onehot vectors
y_one_hot_train=create_one_hot(train_y);
y_one_hot_test=create_one_hot(test_y);

results = @btime train(net, train_x, test_x, y_one_hot_train, y_one_hot_test, mb_size, learning_rate, epochs);
