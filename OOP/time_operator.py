from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from OOP.boundary_conditions import BoundaryCondition, PeriodicBoundary


State = np.ndarray
RHS = Callable[[State], State]
StateFilter = Callable[[State], State]
StepHook = Callable[[int, State], State]


@dataclass(frozen=True)
class SSPRK3:
    """Third-order Runge-Kutta step used in the current scripts."""

    def step(self, state: State, rhs: RHS, dt: float, clean: StateFilter | None = None) -> State:
        clean_state = clean if clean is not None else (lambda value: value)
        q0 = state
        q1 = clean_state(q0 + dt * rhs(q0))
        q2 = clean_state(0.75 * q0 + 0.25 * (q1 + dt * rhs(q1)))
        return clean_state((1.0 / 3.0) * q0 + (2.0 / 3.0) * (q2 + dt * rhs(q2)))


@dataclass
class TimeOperator:
    """Minimal time loop around a spatial RHS."""

    state: State
    rhs: RHS
    dt: float
    boundary: BoundaryCondition = field(default_factory=PeriodicBoundary)
    integrator: SSPRK3 = field(default_factory=SSPRK3)
    state_filter: StateFilter | None = None
    step_hooks: list[StepHook] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.state = self.clean(self.state)

    def clean(self, state: State) -> State:
        if self.state_filter is not None:
            state = self.state_filter(state)
        return self.boundary.apply(state)

    def step(self, step_index: int) -> State:
        self.state = self.integrator.step(self.state, self.rhs, self.dt, clean=self.clean)
        for hook in self.step_hooks:
            self.state = self.clean(hook(step_index, self.state))
        return self.state

    def run(self, t_end: float, save_every: int | None = None) -> tuple[np.ndarray, list[State]]:
        n_steps = int(np.ceil(t_end / self.dt))
        if n_steps > 0:
            self.dt = t_end / n_steps
        if save_every is None:
            save_every = n_steps

        times: list[float] = [0.0]
        states: list[State] = [np.copy(self.state)]
        for step_index in range(n_steps):
            self.step(step_index)
            if (step_index + 1) % save_every == 0 or step_index + 1 == n_steps:
                times.append((step_index + 1) * self.dt)
                states.append(np.copy(self.state))
        return np.array(times), states
