include("../utils/ok_utils.jl")
using BenchmarkTools
using LinearAlgebra
using MLDatasets
using DataFrames
using CSV

# Requires MLDatasets for training

#input data pictures with size 28x28 px = 784
#ouput data value from 0 to 9
#hidden layer of 26 neurons

#Loading data
train_x, train_y = MNIST.traindata(Float64);
test_x, test_y = MNIST.testdata(Float64);
#Creating onehot vectors
y_one_hot_train=create_one_hot(train_y);
y_one_hot_test=create_one_hot(test_y);
train_size = size(train_y,1)
println(train_size)

hidden_neurons = 50
mb_size = 1
learning_rate = 0.01
epochs = 10
tests_number = 5
d = Dict{Any, Any}("epochs" => collect(range(1,epochs)))

for i=1:tests_number
	println("Test $i")

	net=Dict("Layers"=>[], "Pre-Activation"=>[], "Post-Activation"=>[])
	layers=[]
	append!(layers, [fullyconnected(784, hidden_neurons,ReLU())])
	append!(layers, [fullyconnected(hidden_neurons, 10, Softmax())])
	net["Layers"]=layers;

	results = train(net, train_x, test_x, y_one_hot_train, y_one_hot_test, mb_size, learning_rate, epochs, train_size);

	d["train_loss_$i"] = results["train_loss"]
	d["test_loss_$i"] = results["test_loss"]
	d["train_acc_$i"] = results["train_acc"]
	d["test_acc_$i"] = results["test_acc"]
end

df = DataFrame(d)
CSV.write("test4/test4_ok.csv", df)
