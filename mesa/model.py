"""The model class for Mesa framework.

Core Objects: Model
"""

# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import random
import sys
from collections.abc import Sequence

# mypy
from typing import Any

import numpy as np

from mesa.agent import Agent, AgentSet
from mesa.mesa_logging import create_module_logger, method_logger

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


_mesa_logger = create_module_logger()


class Model:
    """Base class for models in the Mesa ABM library.

    This class serves as a foundational structure for creating agent-based models.
    It includes the basic attributes and methods necessary for initializing and
    running a simulation model.

    Attributes:
        running: A boolean indicating if the model should continue running.
        steps: the number of times `model.step()` has been called.
        random: a seeded python.random number generator.
        rng : a seeded numpy.random.Generator

    Notes:
        Model.agents returns the AgentSet containing all agents registered with the model. Changing
        the content of the AgentSet directly can result in strange behavior. If you want change the
        composition of this AgentSet, ensure you operate on a copy.

    """

    @method_logger(__name__)
    def __init__(
        self,
        *args: Any,
        seed: float | None = None,
        rng: RNGLike | SeedLike | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new model with a unique ID.

        Args:
            args: Arbitrary parameters passed on to super class.
            seed: A seed for the random number generator used by the model.
                 Specifying seed is superseded by the parameter rng.
            rng: A random number generator or seed. If not specified, a new Generator is created.
            kwargs: Arbitrary parameters passed on to super class.
        """
        # call init on super class
        super().__init__(*args, **kwargs)

        if rng is None:
            # Setup random number generators
            if seed is None:
                # numpy doesn't understand None as a seed
                seed = random.randrange(sys.maxsize)
                self.random = random.Random(seed)

            else:
                self.random = random.Random(seed)

            self._seed = seed  # this allows for reproducing stdlib.random

            try:
                self.rng: np.random.Generator = np.random.default_rng(seed)
            except TypeError:
                rng = self.random.randint(0, sys.maxsize)
                self.rng: np.random.Generator = np.random.default_rng(rng)
            self._rng = self.rng.bit_generator.state

        else:
            # user specified rng
            if isinstance(rng, np.random.Generator):
                self.rng = rng
            else:
                self.rng = np.random.default_rng(rng)

            # seed parameter is ignored
            self._rng = self.rng.bit_generator.state

            # Seed the stdlib random based on numpy.random
            # so that both of them have the same random numbers
            # The conversion essentially uses numpy to generate
            # a seed for stdlib.random based on the user's provided
            # seed/rng.
            stdlib_seed = self.rng.integers(0, sys.maxsize)
            self.random = random.Random(stdlib_seed)

        # initialization
        self.running = True
        self.steps = 0

        # Wrap the user-defined step method
        self._user_step = self.step
        self.step = self._wrapped_step

        # setup agent registration data structures
        self._agents = {}  # the hard references to all agents in the model
        self._agents_by_type: dict[
            type[Agent], AgentSet
        ] = {}  # a dict with an agentset for each class of agents
        self._all_agents = AgentSet(
            [], random=self.random
        )  # an agentset with all agents

    @property
    def agents(self) -> AgentSet:
        """Return an agent set with all agents in the model.

        Warning:
            Do not set this property!

        Returns:
            AgentSet: An AgentSet containing all agents in the model.

        Raises:
            AttributeError: If you try to set this property.
        """
        return self._all_agents

    @agents.setter
    def agents(self, _):
        raise AttributeError(
            "Setting model.agents is not allowed, since the property `agents` reflects"
            " the internal state of the model. Please provide a different name to store"
            " custom agents."
        )

    @property
    def agents_by_type(self) -> dict[type[Agent], AgentSet]:
        """Return a dictionary of AgentSets, grouped by agent class.

        Returns:
            dict[type[Agent], AgentSet]: A dictionary mapping agent types to AgentSets.
        """
        return self._agents_by_type

    def register_agent(self, agent: Agent) -> None:
        """Add an agent to the model.

        This method registers an agent in the model, adding it to internal data structures.
        The agent is added to the model's agents dictionary and the appropriate AgentSet.

        Args:
            agent: The agent to add.
        """
        # Store the agent in the model's dictionary using the agent's unique ID
        self._agents[agent.unique_id] = agent

        # Get the agent's class
        agent_class = agent.__class__

        # Initialize an AgentSet for this class if one doesn't exist
        if agent_class not in self._agents_by_type:
            self._agents_by_type[agent_class] = AgentSet([], random=self.random)

        # Add the agent to the class-specific AgentSet
        self._agents_by_type[agent_class].add(agent)

        # Add the agent to the global AgentSet
        self._all_agents.add(agent)

    def deregister_agent(self, agent: Agent) -> None:
        """Remove an agent from the model.

        This method removes an agent from the model, removing it from internal data structures.

        Args:
            agent: The agent to remove.
        """
        # Get the agent's unique ID
        uid = agent.unique_id

        # Remove the agent from the _agents dictionary
        if uid in self._agents:
            del self._agents[uid]

        # Get the agent's class
        agent_class = agent.__class__

        # Remove the agent from the class-specific AgentSet if it exists
        if agent_class in self._agents_by_type:
            self._agents_by_type[agent_class].discard(agent)

        # Remove the agent from the global AgentSet
        self._all_agents.discard(agent)

    def get_agent(self, unique_id: Any) -> Agent:
        """Get the agent with the specified ID.

        Args:
            unique_id: The agent's unique ID.

        Returns:
            The agent with the specified ID.

        Raises:
            KeyError: If no agent with the specified ID exists in the model.
        """
        return self._agents[unique_id]

    def remove_all_agents(self) -> None:
        """Remove all agents from the model."""
        # Create a list of agent unique IDs to avoid modifying the dictionary during iteration
        agent_ids = list(self._agents.keys())

        # Remove each agent
        for agent_id in agent_ids:
            # Check if the agent still exists (it might have been removed by another agent's removal)
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.remove()

    @method_logger(__name__)
    def _wrapped_step(self, *args: Any, **kwargs: Any) -> None:
        """Run one step of the model, calling the user-defined step method."""
        # Call the user's step method
        self._user_step(*args, **kwargs)
        self.steps += 1

    def step(self) -> None:
        """Run one step of the model.

        Notes: This is a stub. Override it with User's step method in model.
        Also note that this gets replaced by _wrapped_step during __init__. _wrapped_step
        calls the user-specified step method and then updates the model's `steps` counter.
        """
