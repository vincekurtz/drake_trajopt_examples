#!/usr/bin/env python

##
#
# A simple example of performing swingup control for the pendulum using
# nonlinear MPC.
#
##

from pydrake.all import *

class ModelPredictiveController(LeafSystem):
    """
    MPC controller that solve a simple trajectory optimization problem at a
    lower frequency to define control inputs for the system.
    
    Inputs:
        Current state x

    Outputs:
        Trajectory of control inputs u_nom
        Trajectory of target states x_nom
    """
    def __init__(self):
        LeafSystem.__init__(self)

        # Some options for the trajectory optimization problem
        resolve_period = 0.1
        num_steps = 20
        dt = 0.05

        x_init = np.array([0.0, 0.0])
        x_nom = np.array([3.14, 0.0])

        Q = np.diag([1.0, 0.1])
        R = 1.0*np.eye(1)
        Qf = np.diag([5.0, 0.1])

        # Construct an internal system model
        plant = MultibodyPlant(dt)
        Parser(plant).AddModelFromFile(
                FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
        plant.Finalize()
        context = plant.CreateDefaultContext()
        input_port_index = plant.get_actuation_input_port().get_index()

        # Set up an optimization problem
        optimizer = DirectTranscription(plant, context,
                input_port_index=input_port_index, num_time_samples=num_steps)
        x = optimizer.state()
        u = optimizer.input()
        x0 = optimizer.initial_state()

        x_err = x - x_nom
        optimizer.AddRunningCost( x_err.T@Q@x_err + u.T@R@u )
        optimizer.AddFinalCost( x_err.T@Qf@x_err )

        # Make sure we can modify and re-solve this optimization problem
        self.initial_constraint = optimizer.prog().AddConstraint(eq( x0, x_init))
        self.optimizer = optimizer
        self.prog = optimizer.prog()

        # Set up this system as a discrete-time system
        self.abstract_state = self.DeclareAbstractState(
                AbstractValue.Make(
                    [0.0,                    # start time
                     PiecewisePolynomial(),  # x trajectory
                     PiecewisePolynomial()]  # u trajectory
                    ))
        self.DeclarePeriodicUnrestrictedUpdateEvent(period_sec=resolve_period,
                                                    offset_sec=0.0,
                                                    update=self.SolveTrajOpt)

        # Declare input-output ports
        self.state_input_port = self.DeclareVectorInputPort(
                "state_estimate",
                BasicVector(2))
        self.control_output_port = self.DeclareVectorOutputPort(
                "control_command",
                BasicVector(1),
                self.SendControlCommands)

    def SolveTrajOpt(self, context, state):
        """
        Solve the trajectory optimization problem and store the result in the
        abstract state. 
        """
        print(f"Solving at t={context.get_time()}")

        # Solve the trajectory optimization from the new initial condition
        x_init = self.state_input_port.Eval(context)
        self.initial_constraint.evaluator().UpdateCoefficients(
                Aeq = np.eye(2),
                beq = x_init)
        res = Solve(self.prog)
        assert res.is_success()

        # Put x and u together into a piecewise polynomial trajectory
        ts = self.optimizer.GetSampleTimes(res)
        xs = self.optimizer.GetStateSamples(res)
        us = self.optimizer.GetInputSamples(res)

        start_time = context.get_time()
        x_poly = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(ts, xs)
        u_poly = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(ts, us)
        new_state = [start_time, x_poly, u_poly]

        # Update the abstract state with the optimal trajectory
        state.get_mutable_abstract_state(self.abstract_state).set_value(new_state)

    def SendControlCommands(self, context, output):
        """
        Send control commands computed using a PD controller
        """
        # Get the current time and state estimate
        t = context.get_time()
        x = self.state_input_port.Eval(context)

        # Get the solution from the last traj opt solve
        [t0, x_opt, u_opt] = context.get_abstract_state(self.abstract_state).get_value()

        # N.B. checking t>0 is a hack to ensure that the abstract state is
        # valid. A better way would be to solve the optimization once in the
        # constructor
        if t > 0:
            # Get the nominal state and input for this timestep from the last
            # time we solved the optimization
            x_nom = x_opt.value(t-t0)[:,0]
            u_nom = u_opt.value(t-t0)[0]

            # Compute the input with a simple PD+ controller
            K = np.array([[1.0, 0.1]])
            u = u_nom + K@(x_nom - x)
        else:
            u = np.array([0.0])
        
        output.SetFromVector(u)

if __name__=="__main__":
    # Set up the system diagram
    builder = DiagramBuilder()

    # Pendulum model
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    Parser(plant).AddModelFromFile(
            FindResourceOrThrow("drake/examples/pendulum/Pendulum.urdf"))
    plant.Finalize()

    # MPC controller
    mpc_ctrl = builder.AddSystem(ModelPredictiveController())
    builder.Connect(
            plant.get_state_output_port(),
            mpc_ctrl.state_input_port)
    builder.Connect(
            mpc_ctrl.control_output_port,
            plant.get_actuation_input_port())

    # Visualization
    AddDefaultVisualization(builder)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()

    # Run a simulation
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(5.0)

