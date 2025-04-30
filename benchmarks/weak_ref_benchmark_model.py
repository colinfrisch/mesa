"""Benchmark model for testing the performance of weak references in Mesa."""

import mesa


class WeakRefAgent(mesa.Agent):
    """Agent class for testing weak references.

    This agent maintains references to other agents to test the
    effectiveness of the weak reference implementation.
    """

    def __init__(self, model, connection_count=10):
        """Initialize the agent with its model.

        Args:
            model: The model instance
            connection_count: Number of other agents to connect with
        """
        super().__init__(model)
        self.connection_count = connection_count
        self.connections = []  # Will be filled with references to other agents

    def step(self):
        """Agent's step method that uses its connections."""
        # Perform operation using connections to ensure they're being accessed
        # Store the count of active connections for potential future use
        self.active_connection_count = sum(
            1 for conn in self.connections if conn is not None
        )

        # Occasionally create new connections to replace dead ones
        if len(self.connections) < self.connection_count and self.model.steps % 5 == 0:
            self._add_random_connection()

    def _add_random_connection(self):
        """Add a random connection to another agent."""
        if len(self.model.agents) > 1:  # Make sure there are other agents
            # Get a random agent that isn't self
            possible_agents = [a for a in self.model.agents if a is not self]
            if possible_agents:
                new_connection = self.random.choice(possible_agents)
                self.connections.append(new_connection)


class WeakRefBenchmarkModel(mesa.Model):
    """Model for benchmarking weak reference performance."""

    def __init__(
        self,
        num_agents=100,
        connection_density=0.1,
        agent_turnover=0.05,
        use_weakref=True,
        seed=None,
    ):
        """Initialize the model.

        Args:
            num_agents: Number of agents in the model
            connection_density: Fraction of other agents each agent connects to
            agent_turnover: Fraction of agents that die and get replaced each step
            use_weakref: Whether to use weak references (for comparison)
            seed: Random seed
        """
        super().__init__(seed=seed)

        # Model parameters
        self.num_agents = num_agents
        self.connection_density = connection_density
        self.agent_turnover = agent_turnover
        self.use_weakref = use_weakref

        # Create agents
        for _ in range(self.num_agents):
            connection_count = int(self.num_agents * self.connection_density)
            WeakRefAgent(self, connection_count)

        # Set up initial connections
        self._create_initial_connections()

        # Data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "AgentCount": lambda m: len(m.agents),
            }
        )

    def _create_initial_connections(self):
        """Create initial connections between agents."""
        for agent in self.agents:
            # Calculate how many connections each agent should have
            connection_count = int(self.num_agents * self.connection_density)

            # Create connections
            possible_connections = [a for a in self.agents if a is not agent]
            if len(possible_connections) > 0:
                selected_connections = self.random.sample(
                    possible_connections,
                    min(connection_count, len(possible_connections)),
                )

                agent.connections = selected_connections

    def step(self):
        """Advance the model by one step."""
        # Collect data
        self.datacollector.collect(self)

        # Run agent steps - use AgentSet approach instead of scheduler
        self.agents.shuffle_do("step")

        # Handle agent turnover (death and birth)
        self._agent_turnover()

    def _agent_turnover(self):
        """Remove some agents and add new ones to simulate turnover."""
        # Determine how many agents to remove
        num_to_remove = int(self.num_agents * self.agent_turnover)

        # Select agents to remove
        if num_to_remove > 0 and len(self.agents) > 0:
            # Create a list from the AgentSet for sampling
            agents_list = list(self.agents)
            agents_to_remove = self.random.sample(
                agents_list, min(num_to_remove, len(agents_list))
            )

            # Remove selected agents
            for agent in agents_to_remove:
                agent.remove()

            # Add new agents
            for _ in range(num_to_remove):
                connection_count = int(self.num_agents * self.connection_density)
                new_agent = WeakRefAgent(self, connection_count)

                # Create some initial connections for the new agent
                if len(self.agents) > 1:
                    possible_connections = [
                        a for a in self.agents if a is not new_agent
                    ]
                    connection_count = min(
                        new_agent.connection_count, len(possible_connections)
                    )
                    if connection_count > 0:
                        selected_connections = self.random.sample(
                            possible_connections, connection_count
                        )
                        new_agent.connections = selected_connections
