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

# ╔═╡ 5af7cd80-f262-11ea-2c84-65b966027d9b
begin
	import Pkg
	Pkg.activate(".")
	using ParticleFilters
	using Distributions
	using StaticArrays
	using LinearAlgebra
	using Random
	using PlutoUI
	using Plots
end

# ╔═╡ 44554bc0-f262-11ea-0e76-cbfff70e6c90
md"""
# Using a Particle Filter for Feedback Control

In this tutorial, we will give a brief example of how to use a Particle Filter from `ParticleFilters.jl` for feedback control.

## System Description

The sysem is a two-dimensional discrete-time double integrator with gaussian process and observation noise. The state at time $k$, $x_k$, is a 4-element vector consisting of the position in two dimensions and the velocity. The dynamics can be represented with the linear difference equation:

$$x_{k+1} = f(x_k, u_k, w_k) = A x_k + B u_k + w_k \text{,}$$

where $w_k$ are independent but identically-distributed zero-mean Gaussian random variables with covariance matrix $W$. $A$ and $B$ are given in the code below.

The measurements are noisy observations of the position generated according to 

$$y_{k+1} = h(x_k, u_k, x_{k+1}, v_{k+1}) = C x_{k+1} + v_{k+1} \text{,}$$

where $C$ selects only the position of the model, and $v_k$ are independent identically-distributed zero-mean Gaussian random variables with covariance matrix $V$.

## Control Law Description

The control law, will use the mean from the particle filter belief to try to make the state oscillate about the origin, i.e.

$$u = K \hat{x}$$

where $\hat{x}$ is the mean estimate of the state, and

$$K = [-1, -1, 0, 0]$$

It should be noted that, since this system is linear with Gaussian noise, a [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) would be much better-suited for this case, but we use the system here for its simplicity. In general, particle filters can be used with any nonlinear dynamical systems.
"""

# ╔═╡ 7fefdcea-f262-11ea-05fa-c3732da7e626
md"""
## Dynamics model

We begin by defining the dynamics of the system. For more information about defining a model, see the ["Models" section of the documentation](https://juliapomdp.github.io/ParticleFilters.jl/latest/models/).
"""

# ╔═╡ fcc8f134-f262-11ea-0218-6d4615a3ac2a
const dt = 0.1; # time step

# ╔═╡ 11081f76-f263-11ea-0620-05ed0d0bb51b
const A = [1.0 0.0 dt  0.0;
           0.0 1.0 0.0 dt ;
           0.0 0.0 1.0 0.0;
           0.0 0.0 0.0 1.0];

# ╔═╡ 1109ebd0-f263-11ea-3a0b-f35b84be9eee
const B = [0.5*dt^2 0.0     ;
           0.0      0.5*dt^2;
           dt       0.0     ;
           0.0      dt      ];

# ╔═╡ 111300e4-f263-11ea-3402-678b54969279
const W = Matrix(0.01*Diagonal{Float64}(I, 4)); # Process noise covariance

# ╔═╡ 11138672-f263-11ea-1202-29b59cfdd689
const V = Matrix(Diagonal{Float64}(I, 2)); # Measurement noise covariance

# ╔═╡ 111e49f4-f263-11ea-0a30-01ec97c3819e
f(x, u, rng) = A*x + B*u + rand(rng, MvNormal(W));

# ╔═╡ 2efcb064-f263-11ea-26dc-b77b7549039f
md"""
## Observation Model

Next, the observation model is defined. $h$ generates an observation (this is only used in the simulation, not in the particle filter), $g$ returns the relative likelyhood of an observation given the previous state, control, and current state.
"""

# ╔═╡ 3678ef06-f263-11ea-2528-c3c33b51d692
h(x, rng) = rand(rng, MvNormal(x[1:2], V));

# ╔═╡ 3bcb979c-f263-11ea-35fa-fdbcf12a5da1
g(x0, u, x, y) = pdf(MvNormal(x[1:2], V), y);

# ╔═╡ 47cc1b8e-f263-11ea-1625-3bd91bb0732d
md"""
## Particle Filter

These models are combined to create a model suitable for the filter. Note that the type of the state is designated as a parameter to the constructor. See the ["Models" section of the documentation](https://juliapomdp.github.io/ParticleFilters.jl/latest/models/) for more info.
"""

# ╔═╡ 6953d10c-f263-11ea-0a30-0941e8c7bc84
model = ParticleFilterModel{Vector{Float64}}(f, g);

# ╔═╡ 765a9bc4-f263-11ea-25e1-bf24e1b608aa
@bind N Slider(10:1000; default = 500, show_value = true)

# ╔═╡ 9ca26ce4-f263-11ea-36b3-67ef6dc1db6e
filter = SIRParticleFilter(model, N);

# ╔═╡ a3974042-f263-11ea-3341-cbc37f25e3f9
md"""
## Running a Simulation

To run a simulation, first an initial belief and initial state need to be created. The initial beleif will consist of particles selected uniformly from $[-2, 2]^4$.

The simulation consists of a loop in which the control is calculated from the mean of the particle. Then the state is updated with the dynamics and a new measurement is generated. Finally, the filter is used to update the belief based on this new measurement.

Using the `update` function in this interactive fashion allows the particle  filter to be used as an estimator/observer for feedback control.
"""

# ╔═╡ d3841d84-f263-11ea-1c57-170f949e05cb
begin
	rng = MersenneTwister(1);
	b = ParticleCollection([4.0*rand(4).-2.0 for i in 1:N]); # initial belief
	x = [0.0, 1.0, 1.0, 0.0]; # initial state
	
	@gif for i in 1:100
	    local m = mean(b)
    	u = [-m[1], -m[2]] # Control law - try to orbit the origin
   	 	global x = f(x, u, rng)
    	y = h(x, rng)
    	global b = update(filter, b, u, y)
	
		plt = scatter([p[1] for p in particles(b)], [p[2] for p in particles(b)], 				color=:black, markersize=2, label="")
    	scatter!(plt, [x[1]], [x[2]], color=:blue, xlim=(-5,5), ylim=(-5,5), label="")
	end
end

# ╔═╡ Cell order:
# ╟─44554bc0-f262-11ea-0e76-cbfff70e6c90
# ╠═5af7cd80-f262-11ea-2c84-65b966027d9b
# ╟─7fefdcea-f262-11ea-05fa-c3732da7e626
# ╠═fcc8f134-f262-11ea-0218-6d4615a3ac2a
# ╠═11081f76-f263-11ea-0620-05ed0d0bb51b
# ╠═1109ebd0-f263-11ea-3a0b-f35b84be9eee
# ╠═111300e4-f263-11ea-3402-678b54969279
# ╠═11138672-f263-11ea-1202-29b59cfdd689
# ╠═111e49f4-f263-11ea-0a30-01ec97c3819e
# ╟─2efcb064-f263-11ea-26dc-b77b7549039f
# ╠═3678ef06-f263-11ea-2528-c3c33b51d692
# ╠═3bcb979c-f263-11ea-35fa-fdbcf12a5da1
# ╟─47cc1b8e-f263-11ea-1625-3bd91bb0732d
# ╠═6953d10c-f263-11ea-0a30-0941e8c7bc84
# ╠═765a9bc4-f263-11ea-25e1-bf24e1b608aa
# ╠═9ca26ce4-f263-11ea-36b3-67ef6dc1db6e
# ╟─a3974042-f263-11ea-3341-cbc37f25e3f9
# ╠═d3841d84-f263-11ea-1c57-170f949e05cb
