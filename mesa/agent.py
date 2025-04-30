"""Agent related classes.

Core Objects: Agent and AgentSet.
"""

# Mypy; for the `|` operator purpose
# Remove this __future__ import once the oldest supported Python is 3.10
from __future__ import annotations

import contextlib
import copy
import functools
import itertools
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableSet, Sequence
from random import Random

# mypy
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    # We ensure that these are not imported during runtime to prevent cyclic
    # dependency.
    from mesa.model import Model
    from mesa.space import Position


T = TypeVar("T", bound="Agent")


class WeakAgentRef(Generic[T]):
    """A wrapper class that automatically creates weak references to Agent instances.

    This class acts as a transparent proxy to an Agent instance, automatically creating
    weak references when agents are stored in collections or variables. This helps prevent
    memory leaks and circular references in complex models.

    When an agent is garbage collected, any WeakAgentRef to that agent will automatically
    return None or raise an exception depending on the context.

    Attributes:
        _weak_ref (weakref.ref): The weak reference to the agent

    Notes:
        This is used internally by Mesa to ensure proper memory management. Users
        typically do not need to create WeakAgentRef instances directly.
    """

    def __init__(self, agent: T):
        """Initialize a weak reference to an agent.

        Args:
            agent (Agent): The agent to create a weak reference to
        """
        if isinstance(agent, WeakAgentRef):
            # If already a WeakAgentRef, use its target
            self._weak_ref = agent._weak_ref
        elif isinstance(agent, Agent):
            # Create a new weak reference
            self._weak_ref = weakref.ref(agent)
        else:
            raise TypeError(f"Expected Agent instance, got {type(agent).__name__}")

    def __call__(self) -> T | None:
        """Get the referenced agent or None if it has been garbage collected.

        Returns:
            The agent object or None if it has been garbage collected
        """
        return self._weak_ref()

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the referenced agent.

        Args:
            name: The name of the attribute to access

        Returns:
            The attribute from the referenced agent

        Raises:
            ReferenceError: If the referenced agent has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            raise ReferenceError("Weakly referenced agent no longer exists")
        return getattr(agent, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute assignment to the referenced agent.

        Args:
            name: The name of the attribute to set
            value: The value to set

        Raises:
            ReferenceError: If the referenced agent has been garbage collected
        """
        if name == "_weak_ref":
            # Special case for setting up the weak reference
            super().__setattr__(name, value)
        else:
            agent = self._weak_ref()
            if agent is None:
                raise ReferenceError("Weakly referenced agent no longer exists")
            setattr(agent, name, value)

    def __eq__(self, other: Any) -> bool:
        """Compare the referenced agent with another object.

        Args:
            other: The object to compare with

        Returns:
            True if the referenced agent equals the other object, False otherwise
        """
        if isinstance(other, WeakAgentRef):
            other_agent = other._weak_ref()
            if other_agent is None:
                return False
            other = other_agent

        agent = self._weak_ref()
        if agent is None:
            return False

        return agent == other

    def __hash__(self) -> int:
        """Get the hash of the referenced agent.

        Returns:
            The hash of the referenced agent

        Raises:
            ReferenceError: If the referenced agent has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            raise ReferenceError("Weakly referenced agent no longer exists")
        return hash(agent)

    def __repr__(self) -> str:
        """Get a string representation of the referenced agent.

        Returns:
            A string representation of the referenced agent or an indication that
            it has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            return "<WeakAgentRef: agent has been garbage collected>"
        return f"<WeakAgentRef to {agent!r}>"

    # Support for list and other collection operations
    def __str__(self) -> str:
        """Get a string representation of the referenced agent.

        Returns:
            A string representation of the referenced agent or an indication that
            it has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            return "<Agent has been garbage collected>"
        return str(agent)

    # Support common magic methods by forwarding to the referenced agent
    def __lt__(self, other):
        """Compare if this weakly referenced agent is less than another object.

        Args:
            other: The object to compare with

        Returns:
            bool: Result of the comparison

        Raises:
            ReferenceError: If the referenced agent has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            raise ReferenceError("Weakly referenced agent no longer exists")
        if isinstance(other, WeakAgentRef):
            other = other()
        return agent < other

    def __gt__(self, other):
        """Compare if this weakly referenced agent is greater than another object.

        Args:
            other: The object to compare with

        Returns:
            bool: Result of the comparison

        Raises:
            ReferenceError: If the referenced agent has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            raise ReferenceError("Weakly referenced agent no longer exists")
        if isinstance(other, WeakAgentRef):
            other = other()
        return agent > other

    def __le__(self, other):
        """Compare if this weakly referenced agent is less than or equal to another object.

        Args:
            other: The object to compare with

        Returns:
            bool: Result of the comparison

        Raises:
            ReferenceError: If the referenced agent has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            raise ReferenceError("Weakly referenced agent no longer exists")
        if isinstance(other, WeakAgentRef):
            other = other()
        return agent <= other

    def __ge__(self, other):
        """Compare if this weakly referenced agent is greater than or equal to another object.

        Args:
            other: The object to compare with

        Returns:
            bool: Result of the comparison

        Raises:
            ReferenceError: If the referenced agent has been garbage collected
        """
        agent = self._weak_ref()
        if agent is None:
            raise ReferenceError("Weakly referenced agent no longer exists")
        if isinstance(other, WeakAgentRef):
            other = other()
        return agent >= other


class Agent:
    """Base class for a model agent in Mesa.

    Attributes:
        model (Model): A reference to the model instance.
        unique_id (int): A unique identifier for this agent.
        pos (Position): A reference to the position where this agent is located.

    Notes:
          unique_id is unique relative to a model instance and starts from 1

    """

    # this is a class level attribute
    # it is a dictionary, indexed by model instance
    # so, unique_id is unique relative to a model, and counting starts from 1
    _ids = defaultdict(functools.partial(itertools.count, 1))

    def __init__(self, model: Model, *args, **kwargs) -> None:
        """Create a new agent.

        Args:
            model (Model): The model instance in which the agent exists.
            args: passed on to super
            kwargs: passed on to super

        Notes:
            to make proper use of python's super, in each class remove the arguments and
            keyword arguments you need and pass on the rest to super

        """
        super().__init__(*args, **kwargs)

        self.model: Model = model
        self.unique_id: int = next(self._ids[model])
        self.pos: Position | None = None
        self.model.register_agent(self)

    def get_weak_ref(self) -> WeakAgentRef:
        """Create a weak reference to this agent.

        Returns:
            WeakAgentRef: A weak reference wrapper for this agent
        """
        return WeakAgentRef(self)

    def remove(self) -> None:
        """Remove and delete the agent from the model.

        Notes:
            If you need to do additional cleanup when removing an agent by for example removing
            it from a space, consider extending this method in your own agent class.

        """
        with contextlib.suppress(KeyError):
            self.model.deregister_agent(self)

    def step(self) -> None:
        """A single step of the agent."""

    def advance(self) -> None:  # noqa: D102
        pass

    @classmethod
    def create_agents(cls, model: Model, n: int, *args, **kwargs) -> AgentSet[Agent]:
        """Create N agents.

        Args:
            model: the model to which the agents belong
            args: arguments to pass onto agent instances
                  each arg is either a single object or a sequence of length n
            n: the number of agents to create
            kwargs: keyword arguments to pass onto agent instances
                   each keyword arg is either a single object or a sequence of length n

        Returns:
            AgentSet containing the agents created.

        """

        class ListLike:
            """Helper class to make default arguments act as if they are in a list of length N."""

            def __init__(self, value):
                self.value = value

            def __getitem__(self, i):
                return self.value

        listlike_args = []
        for arg in args:
            if isinstance(arg, (list | np.ndarray | tuple)) and len(arg) == n:
                listlike_args.append(arg)
            else:
                listlike_args.append(ListLike(arg))

        listlike_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (list | np.ndarray | tuple)) and len(v) == n:
                listlike_kwargs[k] = v
            else:
                listlike_kwargs[k] = ListLike(v)

        agents = []
        for i in range(n):
            instance_args = [arg[i] for arg in listlike_args]
            instance_kwargs = {k: v[i] for k, v in listlike_kwargs.items()}
            agent = cls(model, *instance_args, **instance_kwargs)
            agents.append(agent)
        return AgentSet(agents, random=model.random)

    @property
    def random(self) -> Random:
        """Return a seeded stdlib rng."""
        return self.model.random

    @property
    def rng(self) -> np.random.Generator:
        """Return a seeded np.random rng."""
        return self.model.rng


class AgentSet(MutableSet, Sequence):
    """A collection class that represents an ordered set of agents within an agent-based model (ABM).

    This class extends both MutableSet and Sequence, providing set-like functionality with order preservation and
    sequence operations.

    Attributes:
        model (Model): The ABM model instance to which this AgentSet belongs.

    Notes:
        The AgentSet maintains weak references to agents, allowing for efficient management of agent lifecycles
        without preventing garbage collection. It is associated with a specific model instance, enabling
        interactions with the model's environment and other agents. The implementation uses WeakAgentRef
        instances to store agents, which automatically handles weak reference management.

    Notes:
        A `UserWarning` is issued if `random=None`. You can resolve this warning by explicitly
        passing a random number generator. In most cases, this will be the seeded random number
        generator in the model. So, you would do `random=self.random` in a `Model` or `Agent` instance.

    """

    def __init__(self, agents: Iterable[Agent], random: Random | None = None):
        """Initializes the AgentSet with a collection of agents and a reference to the model.

        Args:
            agents (Iterable[Agent]): An iterable of Agent objects to be included in the set.
            random (Random): the random number generator
        """
        if random is None:
            warnings.warn(
                "Random number generator not specified, this can make models non-reproducible. Please pass a random number generator explicitly",
                UserWarning,
                stacklevel=2,
            )
            random = (
                Random()
            )  # FIXME see issue 1981, how to get the central rng from model
        self.random = random
        # Use a standard dictionary instead of WeakKeyDictionary since we're already using WeakAgentRef
        self._agents = {WeakAgentRef(agent): None for agent in agents}

    def __len__(self) -> int:
        """Return the number of agents in the AgentSet."""
        # Filter out any null references before counting
        self._cleanup_null_refs()
        return len(self._agents)

    def _cleanup_null_refs(self):
        """Remove any weak references to agents that have been garbage collected."""
        # Create a new dict without the null references
        self._agents = {ref: None for ref in self._agents if ref() is not None}

    def __iter__(self) -> Iterator[Agent]:
        """Provide an iterator over the agents in the AgentSet."""
        # Clean up before iteration to avoid yielding None
        self._cleanup_null_refs()
        # Return actual agent objects, not the weak references
        return (ref() for ref in self._agents)

    def __contains__(self, agent: Agent) -> bool:
        """Check if an agent is in the AgentSet. Can be used like `agent in agentset`."""
        # Create a weak reference to the agent for comparison
        return any(ref() is agent for ref in self._agents if ref() is not None)

    def select(
        self,
        filter_func: Callable[[Agent], bool] | None = None,
        at_most: int | float = float("inf"),
        inplace: bool = False,
        agent_type: type[Agent] | None = None,
    ) -> AgentSet:
        """Select a subset of agents from the AgentSet based on a filter function and/or quantity limit.

        Args:
            filter_func (Callable[[Agent], bool], optional): A function that takes an Agent and returns True if the
                agent should be included in the result. Defaults to None, meaning no filtering is applied.
            at_most (int | float, optional): The maximum amount of agents to select. Defaults to infinity.
              - If an integer, at most the first number of matching agents are selected.
              - If a float between 0 and 1, at most that fraction of original the agents are selected.
            inplace (bool, optional): If True, modifies the current AgentSet; otherwise, returns a new AgentSet. Defaults to False.
            agent_type (type[Agent], optional): The class type of the agents to select. Defaults to None, meaning no type filtering is applied.

        Returns:
            AgentSet: A new AgentSet containing the selected agents, unless inplace is True, in which case the current AgentSet is updated.

        Notes:
            - at_most just return the first n or fraction of agents. To take a random sample, shuffle() beforehand.
            - at_most is an upper limit. When specifying other criteria, the number of agents returned can be smaller.
        """
        inf = float("inf")
        if filter_func is None and agent_type is None and at_most == inf:
            return self if inplace else copy.copy(self)

        # Check if at_most is of type float
        if at_most <= 1.0 and isinstance(at_most, float):
            at_most = int(len(self) * at_most)  # Note that it rounds down (floor)

        def agent_generator(filter_func, agent_type, at_most):
            count = 0
            for agent_ref in list(self._agents):
                if (agent := agent_ref()) is None:
                    continue

                if count >= at_most:
                    break

                if (not filter_func or filter_func(agent)) and (
                    not agent_type or isinstance(agent, agent_type)
                ):
                    yield agent
                    count += 1

        agents = agent_generator(filter_func, agent_type, at_most)

        return AgentSet(agents, self.random) if not inplace else self._update(agents)

    def shuffle(self, inplace: bool = False) -> AgentSet:
        """Randomly shuffle the order of agents in the AgentSet.

        Args:
            inplace (bool, optional): If True, shuffles the agents in the current AgentSet; otherwise, returns a new shuffled AgentSet. Defaults to False.

        Returns:
            AgentSet: A shuffled AgentSet. Returns the current AgentSet if inplace is True.

        Note:
            Using inplace = True is more performant
        """
        # Clean up null references first
        self._cleanup_null_refs()

        # Get list of WeakAgentRef keys
        weakrefs = list(self._agents)
        self.random.shuffle(weakrefs)

        if inplace:
            self._agents = {ref: None for ref in weakrefs}
            return self
        else:
            return AgentSet(
                (agent for ref in weakrefs if (agent := ref()) is not None), self.random
            )

    def sort(
        self,
        key: Callable[[Agent], Any] | str,
        ascending: bool = False,
        inplace: bool = False,
    ) -> AgentSet:
        """Sort the agents in the AgentSet based on a specified attribute or custom function.

        Args:
            key (Callable[[Agent], Any] | str): A function or attribute name based on which the agents are sorted.
            ascending (bool, optional): If True, the agents are sorted in ascending order. Defaults to False.
            inplace (bool, optional): If True, sorts the agents in the current AgentSet; otherwise, returns a new sorted AgentSet. Defaults to False.

        Returns:
            AgentSet: A sorted AgentSet. Returns the current AgentSet if inplace is True.
        """
        # Clean up null references first
        self._cleanup_null_refs()

        if isinstance(key, str):
            key_attr = key  # Store the attribute name

            # Get the agent from the WeakAgentRef and then get the attribute
            def key_func(ref):
                """Get attribute value from referenced agent if it exists."""
                agent = ref()
                return getattr(agent, key_attr) if agent is not None else None
        else:
            # Wrap the original key function to handle weak references
            original_key = key

            def key_func(ref):
                """Apply original key function to referenced agent if it exists."""
                agent = ref()
                return original_key(agent) if agent is not None else None

        # Sort the weak references based on the key
        sorted_refs = sorted(
            [
                ref for ref in self._agents if ref() is not None
            ],  # Filter out expired refs
            key=key_func,
            reverse=not ascending,
        )

        # Get the actual agents from the sorted weak references
        sorted_agents = [ref() for ref in sorted_refs if ref() is not None]

        return (
            AgentSet(sorted_agents, self.random)
            if not inplace
            else self._update(sorted_agents)
        )

    def _update(self, agents: Iterable[Agent]):
        """Update the AgentSet with a new set of agents.

        This is a private method primarily used internally by other methods like select, shuffle, and sort.
        """
        self._agents = {WeakAgentRef(agent): None for agent in agents}
        return self

    def do(self, method: str | Callable, *args, **kwargs) -> AgentSet:
        """Invoke a method or function on each agent in the AgentSet.

        Args:
            method (str, callable): the callable to do on each agent

                                        * in case of str, the name of the method to call on each agent.
                                        * in case of callable, the function to be called with each agent as first argument

            *args: Variable length argument list passed to the callable being called.
            **kwargs: Arbitrary keyword arguments passed to the callable being called.

        Returns:
            AgentSet | list[Any]: The results of the callable calls if return_results is True, otherwise the AgentSet itself.
        """
        # Clean up null references first
        self._cleanup_null_refs()

        if isinstance(method, str):
            for ref in list(self._agents):
                if (agent := ref()) is not None:
                    getattr(agent, method)(*args, **kwargs)
        else:
            for ref in list(self._agents):
                if (agent := ref()) is not None:
                    method(agent, *args, **kwargs)

        return self

    def shuffle_do(self, method: str | Callable, *args, **kwargs) -> AgentSet:
        """Shuffle the agents in the AgentSet and then invoke a method or function on each agent.

        It's a fast, optimized version of calling shuffle() followed by do().
        """
        # Clean up null references first
        self._cleanup_null_refs()

        weakrefs = list(self._agents)
        self.random.shuffle(weakrefs)

        if isinstance(method, str):
            for ref in weakrefs:
                if (agent := ref()) is not None:
                    getattr(agent, method)(*args, **kwargs)
        else:
            for ref in weakrefs:
                if (agent := ref()) is not None:
                    method(agent, *args, **kwargs)

        return self

    def map(self, method: str | Callable, *args, **kwargs) -> list[Any]:
        """Invoke a method or function on each agent in the AgentSet and return the results.

        Args:
            method (str, callable): the callable to apply on each agent

                                        * in case of str, the name of the method to call on each agent.
                                        * in case of callable, the function to be called with each agent as first argument

            *args: Variable length argument list passed to the callable being called.
            **kwargs: Arbitrary keyword arguments passed to the callable being called.

        Returns:
           list[Any]: The results of the callable calls
        """
        # Clean up null references first
        self._cleanup_null_refs()

        if isinstance(method, str):
            res = [
                getattr(agent, method)(*args, **kwargs)
                for ref in self._agents
                if (agent := ref()) is not None
            ]
        else:
            res = [
                method(agent, *args, **kwargs)
                for ref in self._agents
                if (agent := ref()) is not None
            ]

        return res

    def agg(self, attribute: str, func: Callable) -> Any:
        """Aggregate an attribute of all agents in the AgentSet using a specified function.

        Args:
            attribute (str): The name of the attribute to aggregate.
            func (Callable): The function to apply to the attribute values (e.g., min, max, sum, np.mean).

        Returns:
            Any: The result of applying the function to the attribute values. Often a single value.
        """
        values = self.get(attribute)
        return func(values)

    @overload
    def get(
        self,
        attr_names: str,
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[Any]: ...

    @overload
    def get(
        self,
        attr_names: list[str],
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[list[Any]]: ...

    def get(
        self,
        attr_names,
        handle_missing="error",
        default_value=None,
    ):
        """Retrieve the specified attribute(s) from each agent in the AgentSet.

        Args:
            attr_names (str | list[str]): The name(s) of the attribute(s) to retrieve from each agent.
            handle_missing (str, optional): How to handle missing attributes. Can be:
                                            - 'error' (default): raises an AttributeError if attribute is missing.
                                            - 'default': returns the specified default_value.
            default_value (Any, optional): The default value to return if 'handle_missing' is set to 'default'
                                           and the agent does not have the attribute.

        Returns:
            list[Any]: A list with the attribute value for each agent if attr_names is a str.
            list[list[Any]]: A list with a lists of attribute values for each agent if attr_names is a list of str.

        Raises:
            AttributeError: If 'handle_missing' is 'error' and the agent does not have the specified attribute(s).
            ValueError: If an unknown 'handle_missing' option is provided.
        """
        # Clean up null references first
        self._cleanup_null_refs()

        is_single_attr = isinstance(attr_names, str)
        agents = [ref() for ref in self._agents if ref() is not None]

        if handle_missing == "error":
            if is_single_attr:
                return [getattr(agent, attr_names) for agent in agents]
            else:
                return [
                    [getattr(agent, attr) for attr in attr_names] for agent in agents
                ]

        elif handle_missing == "default":
            if is_single_attr:
                return [getattr(agent, attr_names, default_value) for agent in agents]
            else:
                return [
                    [getattr(agent, attr, default_value) for attr in attr_names]
                    for agent in agents
                ]

        else:
            raise ValueError(
                f"Unknown handle_missing option: {handle_missing}, "
                "should be one of 'error' or 'default'"
            )

    def set(self, attr_name: str, value: Any) -> AgentSet:
        """Set a specified attribute to a given value for all agents in the AgentSet.

        Args:
            attr_name (str): The name of the attribute to set.
            value (Any): The value to set the attribute to.

        Returns:
            AgentSet: The AgentSet instance itself, after setting the attribute.
        """
        for ref in list(self._agents):
            if (agent := ref()) is not None:
                setattr(agent, attr_name, value)
        return self

    def __getitem__(self, item: int | slice) -> Agent:
        """Retrieve an agent or a slice of agents from the AgentSet.

        Args:
            item (int | slice): The index or slice for selecting agents.

        Returns:
            Agent | list[Agent]: The selected agent or list of agents based on the index or slice provided.
        """
        # Clean up null references first
        self._cleanup_null_refs()

        # Get list of actual agents, not weak references
        agents = [ref() for ref in self._agents if ref() is not None]
        return agents[item]

    def add(self, agent: Agent):
        """Add an agent to the AgentSet.

        Args:
            agent (Agent): The agent to add to the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        self._agents[WeakAgentRef(agent)] = None

    def discard(self, agent: Agent):
        """Remove an agent from the AgentSet if it exists.

        This method does not raise an error if the agent is not present.

        Args:
            agent (Agent): The agent to remove from the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        for ref in list(self._agents):
            if ref() is agent:
                del self._agents[ref]
                return

    def remove(self, agent: Agent):
        """Remove an agent from the AgentSet.

        This method raises an error if the agent is not present.

        Args:
            agent (Agent): The agent to remove from the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        for ref in list(self._agents):
            if ref() is agent:
                del self._agents[ref]
                return
        raise KeyError(agent)

    def __getstate__(self):
        """Retrieve the state of the AgentSet for serialization.

        Returns:
            dict: A dictionary representing the state of the AgentSet.
        """
        # Clean up null references first
        self._cleanup_null_refs()

        # Get actual agent objects, not weak references
        agents = [ref() for ref in self._agents if ref() is not None]
        return {"agents": agents, "random": self.random}

    def __setstate__(self, state):
        """Set the state of the AgentSet during deserialization.

        Args:
            state (dict): A dictionary representing the state to restore.
        """
        self.random = state["random"]
        self._update(state["agents"])

    def groupby(self, by: Callable | str, result_type: str = "agentset") -> GroupBy:
        """Group agents by the specified attribute or return from the callable.

        Args:
            by (Callable, str): used to determine what to group agents by

                                * if ``by`` is a callable, it will be called for each agent and the return is used
                                  for grouping
                                * if ``by`` is a str, it should refer to an attribute on the agent and the value
                                  of this attribute will be used for grouping
            result_type (str, optional): The datatype for the resulting groups {"agentset", "list"}

        Returns:
            GroupBy


        Notes:
        There might be performance benefits to using `result_type='list'` if you don't need the advanced functionality
        of an AgentSet.

        """
        # Clean up null references first
        self._cleanup_null_refs()

        groups = defaultdict(list)

        # Get actual agent objects from weak references
        agents = [ref() for ref in self._agents.keys if ref() is not None]

        if isinstance(by, Callable):
            for agent in agents:
                groups[by(agent)].append(agent)
        else:
            for agent in agents:
                groups[getattr(agent, by)].append(agent)

        if result_type == "agentset":
            return GroupBy(
                {k: AgentSet(v, random=self.random) for k, v in groups.items()}
            )
        else:
            return GroupBy(groups)

    # consider adding for performance reasons
    # for Sequence: __reversed__, index, and count
    # for MutableSet clear, pop, remove, __ior__, __iand__, __ixor__, and __isub__


class GroupBy:
    """Helper class for AgentSet.groupby.

    Attributes:
        groups (dict): A dictionary with the group_name as key and group as values

    """

    def __init__(self, groups: dict[Any, list | AgentSet]):
        """Initialize a GroupBy instance.

        Args:
            groups (dict): A dictionary with the group_name as key and group as values

        """
        self.groups: dict[Any, list | AgentSet] = groups

    def map(self, method: Callable | str, *args, **kwargs) -> dict[Any, Any]:
        """Apply the specified callable to each group and return the results.

        Args:
            method (Callable, str): The callable to apply to each group,

                                    * if ``method`` is a callable, it will be called it will be called with the group as first argument
                                    * if ``method`` is a str, it should refer to a method on the group

                                    Additional arguments and keyword arguments will be passed on to the callable.
            args: arguments to pass to the callable
            kwargs: keyword arguments to pass to the callable

        Returns:
            dict with group_name as key and the return of the method as value

        Notes:
            this method is useful for methods or functions that do return something. It
            will break method chaining. For that, use ``do`` instead.

        """
        if isinstance(method, str):
            return {
                k: getattr(v, method)(*args, **kwargs) for k, v in self.groups.items()
            }
        else:
            return {k: method(v, *args, **kwargs) for k, v in self.groups.items()}

    def do(self, method: Callable | str, *args, **kwargs) -> GroupBy:
        """Apply the specified callable to each group.

        Args:
            method (Callable, str): The callable to apply to each group,

                                    * if ``method`` is a callable, it will be called it will be called with the group as first argument
                                    * if ``method`` is a str, it should refer to a method on the group

                                    Additional arguments and keyword arguments will be passed on to the callable.
            args: arguments to pass to the callable
            kwargs: keyword arguments to pass to the callable

        Returns:
            the original GroupBy instance

        Notes:
            this method is useful for methods or functions that don't return anything and/or
            if you want to chain multiple do calls

        """
        if isinstance(method, str):
            for v in self.groups.values():
                getattr(v, method)(*args, **kwargs)
        else:
            for v in self.groups.values():
                method(v, *args, **kwargs)

        return self

    def count(self) -> dict[Any, int]:
        """Return the count of agents in each group.

        Returns:
            dict: A dictionary mapping group names to the number of agents in each group.
        """
        return {k: len(v) for k, v in self.groups.items()}

    def agg(self, attr_name: str, func: Callable) -> dict[Hashable, Any]:
        """Aggregate the values of a specific attribute across each group using the provided function.

        Args:
            attr_name (str): The name of the attribute to aggregate.
            func (Callable): The function to apply (e.g., sum, min, max, mean).

        Returns:
            dict[Hashable, Any]: A dictionary mapping group names to the result of applying the aggregation function.
        """
        return {
            group_name: func([getattr(agent, attr_name) for agent in group])
            for group_name, group in self.groups.items()
        }

    def __iter__(self):  # noqa: D105
        return iter(self.groups.items())

    def __len__(self):  # noqa: D105
        return len(self.groups)
