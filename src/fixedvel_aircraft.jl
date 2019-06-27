# Velocity is fixed but cant be observed. Position changes but can be noisily observed
using ParticleFilters
using Distributions
using StaticArrays
using LinearAlgebra
using Random
using Plots
using Reel
using Statistics

# To calculate the distance between kalman filter mean and true position
function calc_dist(mu::Array,x::Array)
	return norm(mu-x[1:2])
end

# Uses the norm squared calculation function to find the rmse
function calc_rmse(b::ParticleCollection,x)
	norm_vec = calc_norm_squared(b,x)
	return sqrt(mean(norm_vec))
end

"""
Returns an array with each elem being norm squared
from ground truth to particle
"""
function calc_norm_squared(b::ParticleCollection,x)
	particles = b.particles
	n = n_particles(b)	

	norm_squared = zeros(n)
	for i in 1:n
		p = particles[i][1:2]
		norm_squared[i] = norm(p-x[1:2])*norm(p-x[1:2])
	end
	return norm_squared
end

# Calc the rmse of the top 20% particles in the distribution
function calc_rmse_elites(b::ParticleCollection,x)
	particles = b.particles
	n = n_particles(b)
	n_elites = Int(0.2*n)
	norm_vec = calc_norm_squared(b,x)
	elite_particles = particles[sortperm(norm_vec)[1:n_elites]]
	return calc_rmse(ParticleCollection(elite_particles),x)
end

function write_particles_gif(plots,filename)
print("video name = $(filename)")
	frames = Frames(MIME("image/png"), fps=10)
	for plt in plots
	    push!(frames, plt)
	end
	write(filename, frames)
	return nothing
end # End of the reel gif writing function


"""
Q = V measurement noise covariance
R = W process noise covariance
C has to be the correct size to map state to observation
"""
function kalman_filter(mu,sigma,u,z,A,B,C,R,Q)
	mu_bar = A*mu + B*u
	sigma_bar = A*sigma*A' + R

	K = sigma_bar*C'*(inv(C*sigma_bar*C'+Q))

	mu_new = mu_bar+K*(z-C*mu_bar)
	sigma_new = (I-K*C)*sigma_bar
	return mu_new,sigma_new
end

"""
Experiment to run the Kalman filter
"""
function run_kf(mu_0,sig_0,num_iter)
	display("Running kalman filter for $(num_iter) iterations")
	rng = Random.GLOBAL_RNG
	dt = 0.1
	A = [1.0 0.0 dt  0.0;
	     0.0 1.0 0.0 dt ;
	     0.0 0.0 1.0 0.0;
	     0.0 0.0 0.0 1.0]

	B = [0.0 0.0;
	     0.0 0.0;
             0.0  0.0;
             0.0 0.0]

	# Measurement matrix i.e. y = Cx + N(0,V)
	C = [1.0 0.0 0.0 0.0; 
	     0.0 1.0 0.0 0.0]

	W = Matrix(0.001*Diagonal{Float64}(I, 4)) # Process noise covariance
	V = Matrix(5.0*Diagonal{Float64}(I, 2)) # Measurement noise covariance

	f(x, u, rng) = A*x + rand(rng, MvNormal(W))
	h(x, rng) = rand(rng, MvNormal(x[1:2], V)) #Generates an observation	
	mu = mu_0
	sigma = sig_0
	x = [1.,1.,1.,1.]
	u = [-1.0,-1.0] # Just a dummy but still needs to be correct size to multiply B

	plots = []

	for i in 1:num_iter
		x = f(x, u, rng)
		z = h(x, rng)
		mu,sigma = kalman_filter(mu,sigma,u,z,A,B,C,W,V)
		
		plt = scatter([mu[1]], [mu[2]], color=:black, markersize=2.0, label="kf",markershape=:diamond)
		scatter!(plt, [x[1]], [x[2]], color=:blue, xlim=(-5,25), ylim=(-5,25), 
			label = "true")
		push!(plots,plt)
	end
	return plots
end

"""
Run an experiment using all 3 filters
"""
function runexp(;num_particles,num_iter,meascov)
	rng = Random.GLOBAL_RNG

	dt = 0.1 # time step

	A = [1.0 0.0 dt  0.0;
	     0.0 1.0 0.0 dt ;
	     0.0 0.0 1.0 0.0;
	     0.0 0.0 0.0 1.0]

	B = [0.0 0.0;
	     0.0 0.0;
             0.0  0.0;
             0.0 0.0]

		# This will be needed for the Kalman filtering
	C = [1.0 0.0 0.0 0.0; 
	     0.0 1.0 0.0 0.0]

	W = Matrix(0.001*Diagonal{Float64}(I, 4)) # Process noise covariance	
	V = Matrix(meascov*Diagonal{Float64}(I, 2)) # Measurement noise covariance

	f(x, u, rng) = A*x + 0.0*rand(rng, MvNormal(W))

	h(x, rng) = rand(rng, MvNormal(x[1:2], V)) #Generates an observation
	g(x0, u, x, y) = pdf(MvNormal(x[1:2], V), y) #Creates likelihood

	model = ParticleFilterModel{Vector{Float64}}(f, g)

	N = num_particles

	filter_sir = SIRParticleFilter(model, N) # Vanilla particle filter
	filter_cem = CEMParticleFilter(model, N) # CEM filter

		# XXX Initial sampling distribution
	init_dist = Normal(0,5)
	b = ParticleCollection([4.0*rand(init_dist,4).-2.0 for i in 1:N])
	b_cem = b
	
		# XXX: Printing particles
		#print("Initial particle set \n")
		#print_particles(b)
	
		# Parameters setting up the Kalman filtering parameters
	mu = [0.,0.,0.,0.]
	sigma = Matrix(1.0*Diagonal{Float64}(I, 4))


	x = [1.0, 1.0, 1.0, 1.0]

	plots = []

	rmse = zeros(num_iter,3) # Col 1 has vanilla, 2 has cem, 3 has kf
	rmse_elites = zeros(num_iter,2)

	for i in 1:num_iter    #RpB: was 100 before
		#print("\nIteration number $(i) \n")
		m = mean(b) # b is an array of particles. Each element in b is a 4 element tuple
		u = [-m[1], -m[2]] # Control law - try to orbit the origin	
		x = f(x, u, rng)
		y = h(x, rng)
		b = update(filter_sir, b, u, y)
		
		
		b_cem = update(filter_cem,b_cem,u,y)
			# XXX Printing particles			
			#print("\nHere is the cem update\n")
			#print_particles(b_cem)

			# Kalman filter update
		mu,sigma = kalman_filter(mu,sigma,u,y,A,B,C,W,V)

			# Plot the vanilla particle spread
		plt = scatter([p[1] for p in particles(b)], [p[2] for p in particles(b)], color=:cyan, markersize=2.0, label="sir",markershape=:diamond, xlim=(-5,15), ylim=(-5,15))
			
			# Plot the true position
		scatter!(plt, [x[1]], [x[2]], color=:yellow, xlim=(-5,15), ylim=(-5,15), 
			label = "true")
	    
			# Plot the cem particle spread
		scatter!([p[1] for p in particles(b_cem)], [p[2] for p in particles(b_cem)], color=:magenta, markersize=2.0, label="cem",markershape=:cross, xlim=(-5,15), ylim=(-5,15))

			# Plot the kalman filter mean position
		plt = scatter!([mu[1]], [mu[2]], color=:red, markersize=5.0, label="kf",markershape=:octagon,xlim=(-5,15), ylim=(-5,15))

		push!(plots, plt)


		# Plot the rmse value for the current iteration of particles
		# Vanilla rmse
 		rmse_sir=calc_rmse(b,x)
	    	rmse_cem=calc_rmse(b_cem,x)
	    	rmse[i,1] = rmse_sir
	    	rmse[i,2] = rmse_cem

		# Elites calculation
	    	rmse_sir_elites = calc_rmse_elites(b,x)
	    	rmse_cem_elites = calc_rmse_elites(b_cem,x)
	    	rmse_elites[i,1] = rmse_sir_elites
	    	rmse_elites[i,2] = rmse_cem_elites

		# Calculate the distance from kalman filter mean to ground truth
		rmse[i,3] = norm(mu[1:2]-x[1:2])
	end

	#plot(rmse,labels=["sir","cem"])
	#plot(hcat(rmse,rmse_elites),labels=["sir","cem","sir_el","cem_el"])
	#savefig("rmse.png")
	return plots,rmse
	
end # End of the runexp function


# Run the filtering multiple times and average the results from all the experiments
# Third dimension of the `data` data structure denotes experiment number
# Each exp returns a table with timeslices in rows and rmse_sir and rmse_cem
# in columns. Each new table is stacked on top of table from previous experiment
function run_many_exps(;num_exp,num_particles,meascov,num_iter)	
	display("Running $(num_exp) experiments with $(num_particles) particles with $(num_iter) iterations and $(meascov) measurement covariance coeff")	
	data = zeros(num_iter,3,num_exp) #3 columns (vanilla,cem,kf)
	for i in 1:num_exp
		if i%20 == 0.
			print("\nExp num = $(i)\n")
		end		
		plt,data[:,:,i] = runexp(num_particles=num_particles,
					num_iter=num_iter,meascov=meascov)
	end

	rmse_avg = mean(data,dims=3)[:,:,1] #Extract 100x3 array from 100x3x1 array

	plot(rmse_avg,labels=["sir","cem","kf"],xlabel="iteration",ylabel="rmse")
	savefig("../img/25June_$(num_exp)exps_$(num_particles)particles_$(meascov)meascov_$(num_iter)iterations.png")
	return nothing
end

run1exp = true
runmanyexp = false
runkf = false
if run1exp
	# Single experiment and make associated video	
	display("Running a single experiment and making associated video")
	num_particles = 500
	num_iter = 100
	meascov = 5	
	plots, rmse = runexp(num_particles=num_particles,num_iter=num_iter,meascov=meascov)
	@show length(plots) # Should be equal to the number of iterations of the particle filter
	makegif = true
	if makegif write_particles_gif(plots,"../img/26June_$(num_particles)particles_$(num_iter)iterations_$(meascov)meascov.mp4") end
end
if runmanyexp
	# Mulitple experiments to make average rmse plot
	num_particles = 1000
	num_iter = 100
	meascov = 5
	num_exp = 100
	run_many_exps(num_exp = num_exp, num_particles = num_particles,
			num_iter=num_iter,meascov=meascov)
end
if runkf
	mu_0 = [1.,1.,1.,1.]
	sig_0 = Matrix(1.0*Diagonal{Float64}(I, 4))
	num_iter = 500
	
	plot_kf = run_kf(mu_0,sig_0,num_iter)
	makegif = true
	if makegif write_particles_gif(plot_kf,"KalmanFilter_num_iter_$(num_iter).mp4") end
end
