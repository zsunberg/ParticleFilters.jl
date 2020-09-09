### A Pluto.jl notebook ###
# v0.11.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ d9f5114e-f1d9-11ea-3cc9-b5663d73ddea
begin
	import Pkg
	Pkg.activate(".")
	using Distributions
	using Random
	using DelimitedFiles
	using ParticleFilters
	using VegaLite
	using PlutoUI
end

# ╔═╡ 6049db40-f1d9-11ea-24d2-b7193eb2835a
md"""
# Filtering a Preexisting Trajectory or Data Series

In some cases, there is no need to perform control based on the particle belief and instead the goal is to filter a previously-generated sequence of measurements to remove noise or reconstruct hidden state variables. This tutorial will illustrate use of the `runfilter` function for that purpose.
"""

# ╔═╡ c1374e32-f1da-11ea-0a08-97c07f90993d
md"""
## Model

We will use [Euler-integrated](https://en.wikipedia.org/wiki/Euler_method) [Van Der Pol Oscillator Equations](https://en.wikipedia.org/wiki/Van_der_Pol_oscillator) with noise added as the dynamics (`f`), and measurements of the position with gaussian noise added (the pdf is encoded in `g`).

This model is implemented below. For more information on defining models for particle filters, see the [documentation](https://juliapomdp.github.io/ParticleFilters.jl/latest/models/).
"""

# ╔═╡ 818b291c-f1de-11ea-1da8-ed44fc22a749
const dt = 0.2;

# ╔═╡ d4920d88-f210-11ea-0e7e-b161f85ebed7
const mu = 0.8;

# ╔═╡ d492a9d2-f210-11ea-3e4b-e581a6eaff8e
const sigma = 1.0;

# ╔═╡ d4a32bc2-f210-11ea-0d3f-e96d611665f9
function f(x, u, rng)
   	xdot = [x[2], mu*(1-x[1]^2)*x[2] - x[1] + u + 0.1*randn(rng)]
   	return x + dt*xdot
end;

# ╔═╡ d4a3fd18-f210-11ea-3b3f-9be438502d5a
g(x1, u, x2, y) = pdf(Normal(sigma), y-x2[1]);

# ╔═╡ d4b74238-f210-11ea-3f94-0358fef0fbed
 m = ParticleFilterModel{Vector{Float64}}(f, g);

# ╔═╡ b6f414f6-f1de-11ea-1476-6321522d03d9
md"""
## Data

Suppose that the data we would like to filter are contained in text files. [`u.txt`](u.txt) contains the control inputs, and [`y.txt`](y.txt) contains the measurements. The code that generated this data can be found in the [appendix below](#Appendix).
"""

# ╔═╡ c637835a-f1de-11ea-13d4-2f0152a961df
ys = vec(readdlm("y.txt"));

# ╔═╡ da3980e0-f1de-11ea-3fa2-a354a52fa053
us = vec(readdlm("u.txt"));

# ╔═╡ e32811e4-f1de-11ea-186d-876de7f3632f
md"""
## Filtering

We can use an SIR particle filter to get a better estimate of the state. Note that we start with an initial belief consisting of particles with zero velocity and position uniformly distributed in $[-5, 5]$.
"""

# ╔═╡ fbaa2fc2-f1de-11ea-1c4d-37c3f4c7c96f
@bind n Slider(10:1000; default = 500, show_value = true)

# ╔═╡ 050c9b18-f1df-11ea-2037-3d0a836023f9
fil = SIRParticleFilter(m, n);

# ╔═╡ 050d5eca-f1df-11ea-30ea-f15b29c10081
# construct initial belief
b0 = ParticleCollection([[10.0*(rand()-0.5), 0.0] for i in 1:n]);

# ╔═╡ 05166878-f1df-11ea-1071-0bba380df8c7
bs = runfilter(fil, b0, us, ys);

# ╔═╡ 3823e030-f1df-11ea-0ae7-05d0122fc773
md"""
## Plotting the Results

We can now plot the observations, the smoothed estimate of the position (the mean of the belief at every time step), and the true state. Note that the estimated position is much closer to the true position than the noisy measurements.
"""

# ╔═╡ 472e8d38-f1df-11ea-290e-0b6004252868
xmat = readdlm("x.txt");

# ╔═╡ 1bf7e30c-f1ec-11ea-1383-97de6a4392b2
plot_data = [(; t, x1 = x[1], x2 = x[2], y, b_mean = mean(b)[1]) for (t, (x, y, b)) in enumerate(zip(eachslice(xmat; dims=1), ys, bs))]

# ╔═╡ 48422a26-f1ec-11ea-1055-dddfa941851c
plot_data |> @vlplot(x = :t, width = 700) +
@vlplot(:line, y = :x1, color = { datum = "True State" }) +
@vlplot(:point, y = :y, color = { datum = "Observation" }) +
@vlplot(:line, y = :b_mean, color = { datum = "Mean Belief" })

# ╔═╡ 4add757c-f1df-11ea-3e65-1ff81f92161d
md"""
## Appendix


### Code for Generating Data

The following code can be used to generate the data in the text files:

```julia
rng = MersenneTwister(1)

k = 100
h(x, rng) = x[1] + sigma*randn(rng)

x = [3.0, 0.0]
xs = Array{typeof(x)}(undef, k)
us = zeros(k)
ys = zeros(k)
for t in 1:k
    x = f(x, us[t], rng)
    ys[t] = h(x, rng)
    xs[t] = x
end

using DelimitedFiles
writedlm("x.txt", xs)
writedlm("y.txt", ys)
writedlm("u.txt", us)
```
"""

# ╔═╡ Cell order:
# ╟─6049db40-f1d9-11ea-24d2-b7193eb2835a
# ╠═d9f5114e-f1d9-11ea-3cc9-b5663d73ddea
# ╟─c1374e32-f1da-11ea-0a08-97c07f90993d
# ╠═818b291c-f1de-11ea-1da8-ed44fc22a749
# ╠═d4920d88-f210-11ea-0e7e-b161f85ebed7
# ╠═d492a9d2-f210-11ea-3e4b-e581a6eaff8e
# ╠═d4a32bc2-f210-11ea-0d3f-e96d611665f9
# ╠═d4a3fd18-f210-11ea-3b3f-9be438502d5a
# ╠═d4b74238-f210-11ea-3f94-0358fef0fbed
# ╟─b6f414f6-f1de-11ea-1476-6321522d03d9
# ╠═c637835a-f1de-11ea-13d4-2f0152a961df
# ╠═da3980e0-f1de-11ea-3fa2-a354a52fa053
# ╟─e32811e4-f1de-11ea-186d-876de7f3632f
# ╠═fbaa2fc2-f1de-11ea-1c4d-37c3f4c7c96f
# ╠═050c9b18-f1df-11ea-2037-3d0a836023f9
# ╠═050d5eca-f1df-11ea-30ea-f15b29c10081
# ╠═05166878-f1df-11ea-1071-0bba380df8c7
# ╟─3823e030-f1df-11ea-0ae7-05d0122fc773
# ╠═472e8d38-f1df-11ea-290e-0b6004252868
# ╠═1bf7e30c-f1ec-11ea-1383-97de6a4392b2
# ╠═48422a26-f1ec-11ea-1055-dddfa941851c
# ╟─4add757c-f1df-11ea-3e65-1ff81f92161d
