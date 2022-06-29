include("../utils/ns_reverse_accumulation.jl")
include("../utils/ns_optimizer.jl")
using MLDatasets
import Flux: onehotbatch
using BenchmarkTools
using DataFrames
using CSV

function dense(w, b, x, activation) return activation(w * x .+ b) end
function dense(w, x, activation) return activation(w * x) end
function dense(w, x) return w * x end

function cross_entropy_loss(y, ŷ)
    return sum(-y .* log.(ŷ))
end

function relu(x)
    return max.(x, Constant(0))
end

function net(x, wh, wo, bh, bo, y)
    x̂ = dense(wh, bh, x, relu)
    x̂.name = "x̂"
    ŷ = dense(wo, bo, x̂, softmax)
    ŷ.name = "ŷ"
    E = cross_entropy_loss(y, ŷ)
    E.name = "loss"

    return topological_sort(E), ŷ
end

function train(graph :: Vector{Any},
	 		   input_x :: Variable,
			   input_y :: Variable,
			   y_output :: GraphNode,
			   epochs :: Int64,
			   optimizer :: DescentMethod,
			   Wh :: Variable,
			   Wo :: Variable,
			   bh :: Variable,
			   bo :: Variable,
			   train_size :: Int64)
			   
	avg_epoch_train_loss=Vector{Float64}()
    avg_epoch_train_acc=Vector{Float64}()
    avg_epoch_test_loss=Vector{Float64}()
    avg_epoch_test_acc=Vector{Float64}()
	
	for i=1:epochs
		train_loss = 0.
		test_loss = 0.
		train_accuracy = 0.
		test_accuracy = 0.

		println("Epoch: ", i)
		for j in 1:train_size
			@views input_x.output = train_x[j, :]
			@views input_y.output = train_y[j, :]
			train_loss += forward!(graph)
			train_accuracy += argmax(y_output.output)-1 == train_y_outputs[j]
			backward!(graph)
			step!(optimizer, Wh, Wo, bh, bo)
		end

		train_accuracy = train_accuracy / train_size
		train_loss /= train_size
		push!(avg_epoch_train_loss, train_loss)
		push!(avg_epoch_train_acc, train_accuracy)
		
		for j in 1:test_size
			@views input_x.output = test_x[j, :]
			@views input_y.output = test_y[j, :]
			
			test_loss += forward!(graph)
			test_accuracy += argmax(y_output.output)-1 == test_y_outputs[j]
		end
		
		test_accuracy = test_accuracy / test_size
		test_loss /= test_size
		push!(avg_epoch_test_loss, test_loss)
		push!(avg_epoch_test_acc, test_accuracy)
		
		println("Train loss: ", train_loss)
		println("Train accuracy: ", train_accuracy)
		println("Test loss: ", test_loss)
		println("Test accuracy: ", test_accuracy)
		println()
	end
	
	results=Dict("train_loss"=>avg_epoch_train_loss, 
                 "test_loss"=>avg_epoch_test_loss, 
                 "train_acc"=>avg_epoch_train_acc,
                 "test_acc"=>avg_epoch_test_acc)
	return results
end

#Loading data
train_x, train_y = MNIST.traindata(Float64);
test_x, test_y = MNIST.testdata(Float64);

train_x = reshape(train_x, 784, size(train_x, 3))'
test_x = reshape(test_x, 784, size(test_x, 3))'

#Creating onehot vectors
train_y_outputs = train_y
test_y_outputs = test_y

train_y = convert(Matrix{Float64}, onehotbatch(train_y, sort(unique(train_y)))')
test_y = convert(Matrix{Float64}, onehotbatch(test_y, sort(unique(test_y)))')

# train_size = size(train_x, 1)
test_size = size(test_x, 1)

input_neurons  = 784
hidden_neurons = 25
output_neurons = 10
α = 0.01
optimizer = GradientDescent(α)
epochs = 1
train_sizes = [1, 10, 100, 1000, 10000, 35000, 60000]
d = Dict("mean_times" => [],
		 "memory" => [],
		 "allocs" => [],
		 "train_size" => train_sizes,
		 "train_loss" => [],
		 "test_loss" => [],
		 "train_acc" => [],
		 "test_acc" => [])
		 
for train_size=train_sizes
	println("train_size:", train_size)
	Wh  = Variable(randn(hidden_neurons, input_neurons).*0.1, name="wh")
	bh = Variable(zeros(hidden_neurons), name="bh")
	Wo  = Variable(randn(output_neurons, hidden_neurons).*0.1, name="wo")
	bo = Variable(zeros(output_neurons), name="bo")

	input_x = Variable(zeros(784), name="x")
	input_y = Variable(zeros(10), name="y")
	graph, y_output = net(input_x, Wh, Wo, bh, bo, input_y)

	results = @benchmark train($graph, $input_x, $input_y, $y_output, $epochs, $optimizer, $Wh, $Wo, $bh, $bo, $train_size)
	push!(d["mean_times"], mean(results.times))
	push!(d["memory"], results.memory)
	push!(d["allocs"], results.allocs)

	mean_train_loss = 0.0
	mean_test_loss = 0.0
	mean_train_acc = 0.0
	mean_test_acc = 0.0

	test_number = 3
	for i=1:test_number
		Wh  = Variable(randn(hidden_neurons, input_neurons).*0.1, name="wh")
		bh = Variable(zeros(hidden_neurons), name="bh")
		Wo  = Variable(randn(output_neurons, hidden_neurons).*0.1, name="wo")
		bo = Variable(zeros(output_neurons), name="bo")

		input_x = Variable(zeros(784), name="x")
		input_y = Variable(zeros(10), name="y")
		graph, y_output = net(input_x, Wh, Wo, bh, bo, input_y)

		results = train(graph, input_x, input_y, y_output, epochs, optimizer, Wh, Wo, bh, bo, train_size)
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
CSV.write("test3/test3_ns.csv", df)
