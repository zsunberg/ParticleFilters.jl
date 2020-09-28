### A Pluto.jl notebook ###
# v0.11.13

using Markdown
using InteractiveUtils

# ╔═╡ 97400df6-f277-11ea-3e97-77cfa98c1d44
begin
	import Pkg
	Pkg.activate(".")
	using ParticleFilters
	using POMDPs
	using POMDPModels
	using POMDPSimulators
	using POMDPPolicies
	using Random
	using Plots
	using PlutoUI
end

# ╔═╡ 7ccccf4a-f277-11ea-0576-d5ce4ee7905a
md"""
# Using a Particle Filter with POMDPs.jl

The particle filters in `ParticleFilters.jl` can be used out of the box as [updaters](http://juliapomdp.github.io/POMDPs.jl/latest/concepts.html#beliefs_and_updaters-1) for [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl). This tutorial will briefly demonstrate usage with the [LightDark problem from POMDPModels.jl](https://github.com/JuliaPOMDP/POMDPModels.jl/blob/master/src/LightDark.jl).
"""

# ╔═╡ f0763940-f277-11ea-393b-9967e7def5b8
md"""
## Running a Simulation

The following code creates the pomdp model and the associated particle filter and runs a simulation producing a history.
"""

# ╔═╡ f1221eb0-f27a-11ea-1c4d-ad3d49ea5065
rng = MersenneTwister(1)

# ╔═╡ f732e526-f277-11ea-2d6b-15abf9dab2cc
pomdp = LightDark1D();

# ╔═╡ 074584b4-f278-11ea-2236-17eae3937fb6
N = 5000;

# ╔═╡ 07460420-f278-11ea-0e96-e9881f22016e
up = SIRParticleFilter(pomdp, N; rng = rng);

# ╔═╡ 074f4968-f278-11ea-026f-c5ceaf12774e
policy = FunctionPolicy(b->1);

# ╔═╡ 074fceda-f278-11ea-1fe5-59aaa911e05a
b0 = POMDPModels.LDNormalStateDist(-15.0, 5.0);

# ╔═╡ 0757d704-f278-11ea-3f1b-9d26df23e3e9
hr = HistoryRecorder(rng = rng, max_steps=40);

# ╔═╡ 075883ca-f278-11ea-2cbf-0bd4361e434c
history = simulate(hr, pomdp, policy, up, b0);

# ╔═╡ 0d845224-f278-11ea-11cd-73f9c77f6e06
md"""
## Visualization

We can then plot the particle distribution at each step using Plots.jl.

Note that as the belief passes through the light region (centered at y=5)
"""

# ╔═╡ 0ffde4de-f278-11ea-0239-8356ca4e9d2e
@gif for b in belief_hist(history)
	local ys = [s.y for s in particles(b)]
    local nbins = max(1, round(Int, (maximum(ys)-minimum(ys))*2))
    histogram(
		ys,
		xlim = (-20,20),
		ylim = (0,1000),
		nbins = nbins,
		label = "",
		title = "Particle Histogram"
	)                    
end

# ╔═╡ Cell order:
# ╟─7ccccf4a-f277-11ea-0576-d5ce4ee7905a
# ╠═97400df6-f277-11ea-3e97-77cfa98c1d44
# ╟─f0763940-f277-11ea-393b-9967e7def5b8
# ╠═f1221eb0-f27a-11ea-1c4d-ad3d49ea5065
# ╠═f732e526-f277-11ea-2d6b-15abf9dab2cc
# ╠═074584b4-f278-11ea-2236-17eae3937fb6
# ╠═07460420-f278-11ea-0e96-e9881f22016e
# ╠═074f4968-f278-11ea-026f-c5ceaf12774e
# ╠═074fceda-f278-11ea-1fe5-59aaa911e05a
# ╠═0757d704-f278-11ea-3f1b-9d26df23e3e9
# ╠═075883ca-f278-11ea-2cbf-0bd4361e434c
# ╟─0d845224-f278-11ea-11cd-73f9c77f6e06
# ╠═0ffde4de-f278-11ea-0239-8356ca4e9d2e
