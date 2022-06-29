abstract type DescentMethod end

struct GradientDescent <: DescentMethod
  α
end

function step!(O::GradientDescent, parameters...)
    α = O.α
    for parameter in parameters
        parameter.output -= α*parameter.gradient
    end
end