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

train_sizes = [1, 10, 100, 1000, 10000, 35000, 60000]
hidden_neurons = 25
mb_size = 1
learning_rate = 0.01
epochs = 1
d = Dict("mean_times" => [],
		 "memory" => [],
		 "allocs" => [],
		 "train_size" => train_sizes,
		 "train_loss" => [],
		 "test_loss" => [],
		 "train_acc" => [],
		 "test_acc" => [])

for ts=train_sizes
	net=Dict("Layers"=>[], "Pre-Activation"=>[], "Post-Activation"=>[])
	layers=[]
	append!(layers, [fullyconnected(784, hidden_neurons,ReLU())])
	append!(layers, [fullyconnected(hidden_neurons, 10, Softmax())])
	net["Layers"]=layers;

	results = @benchmark train($net, $train_x, $test_x, $y_one_hot_train, $y_one_hot_test, $mb_size, $learning_rate, $epochs, $ts);
	push!(d["mean_times"], mean(results.times))
	push!(d["memory"], results.memory)
	push!(d["allocs"], results.allocs)
	
	mean_train_loss = 0.0
	mean_test_loss = 0.0
	mean_train_acc = 0.0
	mean_test_acc = 0.0

	test_number = 3
	for i=1:test_number
		net=Dict("Layers"=>[], "Pre-Activation"=>[], "Post-Activation"=>[])
		layers=[]
		append!(layers, [fullyconnected(784, hidden_neurons,ReLU())])
		append!(layers, [fullyconnected(hidden_neurons, 10, Softmax())])
		net["Layers"]=layers;

		results = train(net, train_x, test_x, y_one_hot_train, y_one_hot_test, mb_size, learning_rate, epochs, ts);
		mean_train_loss += results["train_loss"][1]
		mean_test_loss += results["test_loss"][1]
		mean_train_acc += results["train_acc"][1]
		mean_test_acc += results["test_acc"][1]
	end
	push!(d["train_loss"], mean_train_loss/test_number)
	push!(d["test_loss"], mean_test_loss/test_number)
	push!(d["train_acc"], mean_train_acc/test_number)
	push!(d["test_acc"], mean_test_acc/test_number)
	
end

df = DataFrame(d)
CSV.write("test3/test3_ok.csv", df)
