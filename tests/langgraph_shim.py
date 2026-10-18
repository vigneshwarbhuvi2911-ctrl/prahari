"""Test-only stand-in for the small part of the LangGraph API this project uses.

It is installed by tests/conftest.py ONLY when the real `langgraph` package is not importable
(e.g. an offline sandbox). The Docker image and the GitHub Actions workflow install real LangGraph,
so the graph is tested against the real library there.
"""
from __future__ import annotations

import sys
import types
import typing

START, END = "__start__", "__end__"


class _Compiled:
    def __init__(self, nodes, edges, cond, reducers):
        self.nodes, self.edges, self.cond, self.reducers = nodes, edges, cond, reducers

    def _run(self, state: dict):
        state = dict(state)
        current = self.edges[START]
        for _ in range(100):
            if current == END:
                return
            update = self.nodes[current](state) or {}
            for k, v in update.items():
                state[k] = self.reducers[k](state.get(k, []), v) if k in self.reducers else v
            yield current, update, state
            if current in self.cond:
                fn, mapping = self.cond[current]
                current = mapping[fn(state)]
            else:
                current = self.edges[current]
        raise RuntimeError("graph did not terminate")

    def invoke(self, state: dict) -> dict:
        final = dict(state)
        for _, _, final in self._run(state):
            pass
        return final

    def stream(self, state: dict, stream_mode: str = "updates"):
        for node, update, _ in self._run(state):
            yield {node: update}


class StateGraph:
    def __init__(self, schema):
        self.nodes, self.edges, self.cond = {}, {}, {}
        self.reducers = {}
        for name, hint in typing.get_type_hints(schema, include_extras=True).items():
            if typing.get_origin(hint) is typing.Annotated:
                self.reducers[name] = hint.__metadata__[0]

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, src, fn, mapping):
        self.cond[src] = (fn, mapping)

    def compile(self):
        return _Compiled(self.nodes, self.edges, self.cond, self.reducers)


def install() -> None:
    pkg = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")
    graph.StateGraph, graph.START, graph.END = StateGraph, START, END
    pkg.graph = graph
    sys.modules["langgraph"] = pkg
    sys.modules["langgraph.graph"] = graph
