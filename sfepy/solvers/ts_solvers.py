"""
Time stepping solvers.
"""
import numpy as nm

from sfepy.base.base import output, Struct, IndexedStruct, basestr
from sfepy.solvers.solvers import SolverMeta, TimeSteppingSolver
from sfepy.solvers.ts import TimeStepper, VariableTimeStepper

class StationarySolver(TimeSteppingSolver):
    """
    Solver for stationary problems without time stepping.

    This class is provided to have a unified interface of the time stepping
    solvers also for stationary problems.
    """
    name = 'ts.stationary'

    __metaclass__ = SolverMeta

    def __init__(self, conf, **kwargs):
        TimeSteppingSolver.__init__(self, conf, ts=None, **kwargs)

    def __call__(self, state0=None, save_results=True, step_hook=None,
                 post_process_hook=None, nls_status=None):
        problem = self.problem

        restart_filename = problem.conf.options.get('load_restart', None)
        if restart_filename is not None:
            state = problem.load_restart(restart_filename, state=state0)

        else:
            state = problem.solve(state0=state0, nls_status=nls_status)

        if step_hook is not None:
            step_hook(problem, None, state)

        restart_filename = problem.get_restart_filename()
        if restart_filename is not None:
            problem.save_restart(restart_filename, state)

        if save_results:
            problem.save_state(problem.get_output_name(), state,
                               post_process_hook=post_process_hook,
                               file_per_var=None)

        yield 0, 0.0, state

    def init_time(self, nls_status=None):
        self.problem.time_update()
        self.problem.init_solvers(nls_status=nls_status)

def replace_virtuals(deps, pairs):
    out = {}
    for key, val in deps.iteritems():
        out[pairs[key]] = val

    return out

class EquationSequenceSolver(TimeSteppingSolver):
    """
    Solver for stationary problems with an equation sequence.
    """
    name = 'ts.equation_sequence'

    __metaclass__ = SolverMeta

    def __init__(self, conf, **kwargs):
        TimeSteppingSolver.__init__(self, conf, ts=None, **kwargs)

    def __call__(self, state0=None, save_results=True, step_hook=None,
                 post_process_hook=None, nls_status=None):
        from sfepy.base.base import invert_dict, get_subdict
        from sfepy.base.resolve_deps import resolve

        problem = self.problem

        if state0 is None:
            state0 = problem.create_state()

        variables = problem.get_variables()
        vtos = variables.get_dual_names()
        vdeps = problem.equations.get_variable_dependencies()
        sdeps = replace_virtuals(vdeps, vtos)

        sorder = resolve(sdeps)

        stov = invert_dict(vtos)
        vorder = [[stov[ii] for ii in block] for block in sorder]

        parts0 = state0.get_parts()
        state = state0.copy()
        solved = []
        for ib, block in enumerate(vorder):
            output('solving for %s...' % sorder[ib])

            subpb = problem.create_subproblem(block, solved)

            subpb.equations.print_terms()

            subpb.time_update()
            substate0 = subpb.create_state()

            vals = get_subdict(parts0, block)
            substate0.set_parts(vals)

            substate = subpb.solve(state0=substate0, nls_status=nls_status)

            state.set_parts(substate.get_parts())

            solved.extend(sorder[ib])
            output('...done')

        if step_hook is not None:
            step_hook(problem, None, state)

        if save_results:
            problem.save_state(problem.get_output_name(), state,
                               post_process_hook=post_process_hook,
                               file_per_var=None)

        yield 0, 0.0, state

    def init_time(self, nls_status=None):
        self.problem.init_solvers(nls_status=nls_status)

def get_initial_state(problem):
    """
    Create a zero state vector and apply initial conditions.
    """
    state = problem.create_state()

    problem.setup_ics()
    state.apply_ic()

    return state

def prepare_save_data(ts, conf):
    """
    Given a time stepper configuration, return a list of time steps when the
    state should be saved.
    """
    try:
        save_steps = conf.options.save_steps
    except:
        save_steps = -1

    if save_steps == -1:
        save_steps = ts.n_step

    is_save = nm.linspace(0, ts.n_step - 1, save_steps).astype(nm.int32)
    is_save = nm.unique(is_save)

    return ts.suffix, is_save

def prepare_matrix(problem, state):
    """
    Pre-assemble tangent system matrix.
    """
    problem.update_materials()

    ev = problem.get_evaluator()
    try:
        mtx = ev.eval_tangent_matrix(state(), is_full=True)

    except ValueError:
        output('matrix evaluation failed, giving up...')
        raise

    return mtx

def make_implicit_step(ts, state0, problem, nls_status=None):
    """
    Make a step of an implicit time stepping solver.
    """
    if ts.step == 0:
        state0.apply_ebc()
        state = state0.copy(deep=True)

        if not ts.is_quasistatic:
            ev = problem.get_evaluator()
            try:
                vec_r = ev.eval_residual(state(), is_full=True)
            except ValueError:
                output('initial residual evaluation failed, giving up...')
                raise
            else:
                err = nm.linalg.norm(vec_r)
                output('initial residual: %e' % err)

        if problem.is_linear():
            mtx = prepare_matrix(problem, state)
            problem.try_presolve(mtx)

        # Initialize variables with history.
        state0.init_history()
        if ts.is_quasistatic:
            # Ordinary solve.
            state = problem.solve(state0=state0, nls_status=nls_status)

    else:
        problem.time_update(ts)
        state = problem.solve(state0=state0, nls_status=nls_status)

    return state

def get_min_dt(adt):
    red = adt.red
    while red >= adt.red_max:
        red *= adt.red_factor

    dt = adt.dt0 * red

    return dt

def adapt_time_step(ts, status, adt, problem=None):
    """
    Adapt the time step of `ts` according to the exit status of the
    nonlinear solver.

    The time step dt is reduced, if the nonlinear solver did not converge. If it
    converged in less then a specified number of iterations for several time
    steps, the time step is increased. This is governed by the following
    parameters:

    - red_factor : time step reduction factor
    - red_max : maximum time step reduction factor
    - inc_factor : time step increase factor
    - inc_on_iter : increase time step if the nonlinear solver converged in
      less than this amount of iterations...
    - inc_wait : ...for this number of consecutive time steps

    Parameters
    ----------
    ts : VariableTimeStepper instance
        The time stepper.
    status : IndexedStruct instance
        The nonlinear solver exit status.
    adt : Struct instance
        The adaptivity parameters of the time solver:
    problem : Problem instance, optional
        This canbe used in user-defined adaptivity functions. Not used here.

    Returns
    -------
    is_break : bool
        If True, the adaptivity loop should stop.
    """
    is_break = False

    if status.condition == 0:
        if status.n_iter <= adt.inc_on_iter:
            adt.wait += 1

            if adt.wait > adt.inc_wait:
                if adt.red < 1.0:
                    adt.red = adt.red * adt.inc_factor
                    ts.set_time_step(adt.dt0 * adt.red)
                    output('+++++ new time step: %e +++++' % ts.dt)
                adt.wait = 0

        else:
            adt.wait = 0

        is_break = True

    else:
        adt.red = adt.red * adt.red_factor
        if adt.red < adt.red_max:
            is_break = True

        else:
            ts.set_time_step(adt.dt0 * adt.red, update_time=True)
            output('----- new time step: %e -----' % ts.dt)
            adt.wait = 0

    return is_break

class SimpleTimeSteppingSolver(TimeSteppingSolver):
    """
    Implicit time stepping solver with a fixed time step.
    """
    name = 'ts.simple'

    __metaclass__ = SolverMeta

    _parameters = [
        ('t0', 'float', 0.0, False,
         'The initial time.'),
        ('t1', 'float', 1.0, False,
         'The final time.'),
        ('dt', 'float', None, False,
         'The time step. Used if `n_step` is not given.'),
        ('n_step', 'int', 10, False,
         'The number of time steps. Has precedence over `dt`.'),
        ('quasistatic', 'bool', False, False,
         """If True, assume a quasistatic time-stepping. Then the non-linear
            solver is invoked also for the initial time."""),
    ]

    def __init__(self, conf, **kwargs):
        TimeSteppingSolver.__init__(self, conf, **kwargs)

        self.ts = TimeStepper.from_conf(self.conf)

        nd = self.ts.n_digit
        format = '====== time %%e (step %%%dd of %%%dd) =====' % (nd, nd)

        self.format = format

    def __call__(self, state0=None, save_results=True, step_hook=None,
                 post_process_hook=None, nls_status=None):
        """
        Solve the time-dependent problem.
        """
        problem = self.problem
        ts = self.ts

        suffix, is_save = prepare_save_data(ts, problem.conf)

        if state0 is None:
            state0 = get_initial_state(problem)

        restart_filename = problem.conf.options.get('load_restart', None)
        if restart_filename is not None:
            state0.init_history()
            state0 = problem.load_restart(restart_filename, state=state0, ts=ts)
            problem.advance(ts)
            ts.advance()

        ii = 0 # Broken with restart.
        for step, time in ts.iter_from(ts.step):
            output(self.format % (time, step + 1, ts.n_step))

            state = self.solve_step(ts, state0, nls_status=nls_status)
            state0 = state.copy(deep=True)

            if step_hook is not None:
                step_hook(problem, ts, state)

            restart_filename = problem.get_restart_filename(ts=ts)
            if restart_filename is not None:
                problem.save_restart(restart_filename, state, ts=ts)

            if save_results and (is_save[ii] == ts.step):
                filename = problem.get_output_name(suffix=suffix % ts.step)
                problem.save_state(filename, state,
                                   post_process_hook=post_process_hook,
                                   file_per_var=None,
                                   ts=ts)
                ii += 1

            yield step, time, state

            problem.advance(ts)

    def init_time(self, nls_status=None):
        ts = self.ts
        problem = self.problem

        problem.time_update(ts)
        problem.init_solvers(nls_status=nls_status)

        if not ts.is_quasistatic:
            problem.init_time(ts)

    def solve_step(self, ts, state0, nls_status=None):
        """
        Solve a single time step.
        """
        state = make_implicit_step(ts, state0, self.problem,
                                   nls_status=nls_status)

        return state

class AdaptiveTimeSteppingSolver(SimpleTimeSteppingSolver):
    """
    Implicit time stepping solver with an adaptive time step.

    Either the built-in or user supplied function can be used to adapt the time
    step.
    """
    name = 'ts.adaptive'

    __metaclass__ = SolverMeta

    _parameters = SimpleTimeSteppingSolver._parameters + [
        ('adapt_fun', 'callable(ts, status, adt, problem)', None, False,
         """If given, use this function to set the time step in `ts`. The
            function return value is a bool - if True, the adaptivity loop
            should stop. The other parameters below are collected in `adt`,
            `status` is the nonlinear solver status and `problem` is the
            :class:`Problem <sfepy.discrete.problem.Problem>` instance."""),
        ('dt_red_factor', 'float', 0.2, False,
         'The time step reduction factor.'),
        ('dt_red_max', 'float', 1e-3, False,
         'The maximum time step reduction factor.'),
        ('dt_inc_factor', 'float', 1.25, False,
         'The time step increase factor.'),
        ('dt_inc_on_iter', 'int', 4, False,
         """Increase the time step if the nonlinear solver converged in less
            than this amount of iterations for `dt_inc_wait` consecutive time
            steps."""),
        ('dt_inc_wait', 'int', 5, False,
         'The number of consecutive time steps, see `dt_inc_on_iter`.'),
    ]

    def __init__(self, conf, **kwargs):
        TimeSteppingSolver.__init__(self, conf, **kwargs)

        self.ts = VariableTimeStepper.from_conf(self.conf)

        get = self.conf.get
        adt = Struct(red_factor=get('dt_red_factor', 0.2),
                     red_max=get('dt_red_max', 1e-3),
                     inc_factor=get('dt_inc_factor', 1.25),
                     inc_on_iter=get('dt_inc_on_iter', 4),
                     inc_wait=get('dt_inc_wait', 5),
                     red=1.0, wait=0, dt0=0.0)
        self.adt = adt

        adt.dt0 = self.ts.get_default_time_step()
        self.ts.set_n_digit_from_min_dt(get_min_dt(adt))

        self.format = '====== time %e (dt %e, wait %d, step %d of %d) ====='

        if isinstance(self.conf.adapt_fun, basestr):
            self.adapt_time_step = self.problem.functions[self.conf.adapt_fun]

        else:
            self.adapt_time_step = self.conf.adapt_fun

    def __call__(self, state0=None, save_results=True, step_hook=None,
                 post_process_hook=None, nls_status=None):
        """
        Solve the time-dependent problem.
        """
        problem = self.problem
        ts = self.ts

        if state0 is None:
            state0 = get_initial_state(problem)

        restart_filename = problem.conf.options.get('load_restart', None)
        if restart_filename is not None:
            state0.init_history()
            state0 = problem.load_restart(restart_filename, state=state0, ts=ts)
            problem.advance(ts)
            ts.advance()

        for step, time in ts.iter_from_current():
            output(self.format % (time, ts.dt, self.adt.wait,
                                  step + 1, ts.n_step))

            state = self.solve_step(ts, state0, nls_status=nls_status)
            state0 = state.copy(deep=True)

            if step_hook is not None:
                step_hook(problem, ts, state)

            restart_filename = problem.get_restart_filename(ts=ts)
            if restart_filename is not None:
                problem.save_restart(restart_filename, state, ts=ts)

            if save_results:
                filename = problem.get_output_name(suffix=ts.suffix % ts.step)
                problem.save_state(filename, state,
                                   post_process_hook=post_process_hook,
                                   file_per_var=None,
                                   ts=ts)

            yield step, time, state

            problem.advance(ts)

    def solve_step(self, ts, state0, nls_status=None):
        """
        Solve a single time step.
        """
        status = IndexedStruct(n_iter=0, condition=0)
        while 1:
            state = make_implicit_step(ts, state0, self.problem,
                                       nls_status=status)

            is_break = self.adapt_time_step(ts, status, self.adt, self.problem)
            if is_break:
                break

        if nls_status is not None:
            nls_status.update(status)

        return state
