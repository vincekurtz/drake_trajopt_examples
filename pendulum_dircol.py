#!/usr/bin/env python

##
#
# Solve the swing-up control problem for the pendulum using direct collocation,
# where instead of considering a discrete-time system, we consider a
# continuous-time system at certain knot points.
#
# See http://underactuated.mit.edu/trajopt.html#section3.
#
##

import numpy as np
from pydrake.all import *
import time

# Problem definition
num_knots = 20
min_dt = 0.05
max_dt = 0.5
T = 2.0

x_init = np.array([0.0, 0.0])
x_nom = np.array([np.pi, 0])

Q = np.diag([0.0, 0.1])
R = 1.0*np.eye(1)
Qf = np.diag([100,1])

# Create a plant model
plant = MultibodyPlant(0)
Parser(plant).AddModelFromFile(
        FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
plant.Finalize()

context = plant.CreateDefaultContext()
input_port_index = plant.get_actuation_input_port().get_index()

# Set up the solver
optimizer = DirectCollocation(plant, context,
                        input_port_index=input_port_index,
                        num_time_samples=num_knots,
                        minimum_timestep=min_dt,
                        maximum_timestep=max_dt)

x = optimizer.state()
u = optimizer.input()
x0 = optimizer.initial_state()

optimizer.prog().AddConstraint(eq( x0, x_init ))
x_err = x - x_nom
optimizer.AddRunningCost( x_err.T@Q@x_err + u.T@R@u )
optimizer.AddFinalCost( x_err.T@Qf@x_err )

optimizer.AddDurationBounds(T,T)

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
states = optimizer.ReconstructStateTrajectory(res)
knot_times = optimizer.GetSampleTimes(res)

# Play back the solution on drake-visualizer
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0)
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

dt = 1e-2
timesteps = np.arange(0,T,dt)
for t in timesteps:
    x = states.value(t)

    diagram_context.SetTime(t)
    plant.SetPositionsAndVelocities(plant_context, x)
    diagram.Publish(diagram_context)

    time.sleep(dt)
