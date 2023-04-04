#!/usr/bin/env python

##
#
# Swing-up control of a pendulum. 
#
# This solves the problem
#
#   min_{x,u} x_err[T]'*Qf*x_err[T] +
#                  dt*sum_{t=0}^{T-1} x_err[t]'*Q*x_err[t] + u[t]'*R*u[t]
#   s.t. x[t+1] = f(x[t], u[t])
#        x[0] = x_init
#
# where f(x,u) defines the discrete-time dynamics of the pendulum, 
# x = [theta, theta_dot] is the pendulum state, u[t] are joint torques at time
# t, and x_err[t] = x[t] - x_nom is an error w.r.t. some nominal state. 
#
##

import numpy as np
from pydrake.all import *
import time

# Problem definition
num_steps = 40
dt = 5e-2

x_init = np.array([0.0, 0.0])
x_nom = np.array([np.pi, 0])

Q = np.diag([0.0, 0.1])
R = 1.0*np.eye(1)
Qf = np.diag([100,1])

# Create a plant model
plant = MultibodyPlant(dt)
Parser(plant).AddModelFromFile(
        FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
plant.Finalize()

context = plant.CreateDefaultContext()
input_port_index = plant.get_actuation_input_port().get_index()

# Set up the solver
optimizer = DirectTranscription(plant, context,
        input_port_index=input_port_index, num_time_samples=num_steps)

x = optimizer.state()
u = optimizer.input()
x0 = optimizer.initial_state()

optimizer.prog().AddConstraint(eq( x0, x_init ))
x_err = x - x_nom
optimizer.AddRunningCost( x_err.T@Q@x_err + u.T@R@u )
optimizer.AddFinalCost( x_err.T@Qf@x_err )

# Solve the optimization problem
solver = SnoptSolver()
#solver = IpoptSolver()

start_time = time.time()
res = solver.Solve(optimizer.prog())
solve_time = time.time() - start_time

solver_name = res.get_solver_id().name()
optimal_cost = res.get_optimal_cost()
if not res.is_success():
    print("Solver failure reported!")
print(f"Solved in {solve_time} seconds using {solver_name}")
print(f"Optimal cost: {optimal_cost}")

# Extract the solution
timesteps = optimizer.GetSampleTimes(res)
states = optimizer.GetStateSamples(res)
inputs = optimizer.GetInputSamples(res)

# Play back the solution on drake-visualizer
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, dt)
Parser(plant).AddModelFromFile(
        FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
plant.Finalize()

zero_input = builder.AddSystem(ConstantVectorSource(np.zeros(1)))
builder.Connect(
        zero_input.get_output_port(),
        plant.get_actuation_input_port())

DrakeVisualizer().AddToBuilder(builder, scene_graph)

diagram = builder.Build()
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

for i in range(len(timesteps)):
    t = timesteps[i]
    x = states[:,i]

    diagram_context.SetTime(t)
    plant.SetPositionsAndVelocities(plant_context, x)
    diagram.ForcedPublish(diagram_context)

    time.sleep(dt)

