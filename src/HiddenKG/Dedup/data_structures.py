from dataclasses import dataclass
from typing import List

@dataclass
class Occurrence:
    path: str
    node_id: str
    level: int
    title: str

@dataclass
class Neighbor:
    name: str
    snippet: str = ""  # 例 "entity_related|undirected" 或 "has_subordinate|out"

@dataclass
class EntityItem:
    name: str
    alias: List[str]
    type: str
    original: str
    updated_description: str
    role: str
    occurrences: List[Occurrence]
    neighbors: List[Neighbor]