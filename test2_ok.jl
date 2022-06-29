include("ok_utils.jl")
using BenchmarkTools
using LinearAlgebra
using MLDatasets
using DataFrames
using CSV
import Random: seed!
seed!(0)

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

batch_size = [1, 10, 50, 100, 1000, 10000, 35000, 60000]
hidden_neurons = 25
learning_rate = 0.01
epochs = 1
d = Dict("mean_times" => [],
		 "memory" => [],
		 "allocs" => [],
		 "batch_size" => batch_size,
		 "train_loss" => [],
		 "test_loss" => [],
		 "train_acc" => [],
		 "test_acc" => [])

for bs=batch_size
	net=Dict("Layers"=>[], "Pre-Activation"=>[], "Post-Activation"=>[])
	layers=[]
	append!(layers, [fullyconnected(784, hidden_neurons,ReLU())])
	append!(layers, [fullyconnected(hidden_neurons, 10, Softmax())])
	net["Layers"]=layers;

	results = @benchmark train($net, $train_x, $test_x, $y_one_hot_train, $y_one_hot_test, $bs, $learning_rate, $epochs, $train_size);
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

		results = train(net, train_x, test_x, y_one_hot_train, y_one_hot_test, bs, learning_rate, epochs, train_size);
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
CSV.write("test2/test2_ok.csv", df)
