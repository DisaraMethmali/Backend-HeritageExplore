"""
causal_knowledge_graph.py
=========================
Full Causal Knowledge Graph + Contradiction Engine for Sri Lanka Historical Chatbot.

Replaces the basic CausalChainEngine with:
  - NetworkX directed acyclic graph of historical facts/events
  - Temporal constraint solver (year-tagged nodes)
  - Anachronism detection (character knowledge window enforcement)
  - Causal path inference (multi-hop reasoning)
  - Contradiction detection (false premises, wrong causality)
  - Live claim validation against the graph
  - Flask endpoint integration  (/causal/*)
  - Flutter-ready JSON API

Usage:
    from causal_knowledge_graph import CausalKnowledgeGraph, ContradictionEngine

    graph  = CausalKnowledgeGraph()
    engine = ContradictionEngine(graph)

    # In your generate_answer pipeline:
    validation = engine.validate_claim(query, character_id)
    chain      = graph.get_causal_path("dutch_arrival", "british_takeover")
"""

from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    print("WARNING: networkx not installed. Run: pip install networkx")

# ─────────────────────────────────────────────────────────────────────────────
#  GRAPH DATA  —  nodes are historical facts/events, edges are causal/temporal
# ─────────────────────────────────────────────────────────────────────────────

# Each node: id, label, year_from, year_to, description, characters_aware, tags
# Each edge: source→target, relation (CAUSES | ENABLES | PRECEDES | CONTRADICTS)

GRAPH_NODES = [
    # ── Buddhism & the Tooth Relic ───────────────────────────────────────────
    {
        "id": "buddha_death",
        "label": "Death of the Buddha",
        "year_from": -483, "year_to": -483,
        "description": "Siddhartha Gautama passed into Parinirvana; his remains including a tooth were preserved.",
        "characters_aware": ["citizen"],
        "tags": ["buddhism", "relic", "india"],
    },
    {
        "id": "relic_india",
        "label": "Tooth Relic kept in Kalinga (India)",
        "year_from": -483, "year_to": 313,
        "description": "The Sacred Tooth Relic was venerated in Kalinga, India for nearly eight centuries.",
        "characters_aware": ["citizen"],
        "tags": ["relic", "buddhism", "india"],
    },
    {
        "id": "relic_arrives_lanka",
        "label": "Tooth Relic arrives in Sri Lanka",
        "year_from": 313, "year_to": 313,
        "description": "Princess Hemamala smuggled the Tooth Relic to Sri Lanka hidden in her hair.",
        "characters_aware": ["king", "nilame", "citizen"],
        "tags": ["relic", "buddhism", "kandy", "landmark"],
    },
    {
        "id": "relic_legitimises_rule",
        "label": "Relic as symbol of royal legitimacy",
        "year_from": 313, "year_to": 1815,
        "description": "Every king who possessed the Tooth Relic was recognised as the rightful Buddhist ruler.",
        "characters_aware": ["king", "nilame", "citizen"],
        "tags": ["relic", "kingship", "buddhism", "kandy"],
    },
    {
        "id": "dalada_maligawa_built",
        "label": "Temple of the Tooth established in Kandy",
        "year_from": 1592, "year_to": 1592,
        "description": "Sri Dalada Maligawa was established in Kandy, making Kandy the permanent custodian of the Relic.",
        "characters_aware": ["king", "nilame", "citizen"],
        "tags": ["temple", "kandy", "relic", "landmark"],
    },
    {
        "id": "esala_perahera_origin",
        "label": "Esala Perahera tradition established",
        "year_from": -200, "year_to": -200,
        "description": "The Esala Perahera procession tradition began; it has been held annually for over 2,000 years.",
        "characters_aware": ["nilame", "citizen"],
        "tags": ["festival", "buddhism", "temple"],
    },
    {
        "id": "perahera_replica_not_real",
        "label": "Perahera carries replica casket, NOT the actual Relic",
        "year_from": 1592, "year_to": 9999,
        "description": (
            "During the Esala Perahera, the procession elephant carries an ornate golden "
            "reliquary REPLICA. The actual Sacred Tooth Relic remains locked in the inner "
            "shrine of Sri Dalada Maligawa throughout the festival."
        ),
        "characters_aware": ["nilame", "citizen"],
        "tags": ["festival", "relic", "temple", "misconception_target"],
    },

    # ── Kandyan Kingdom ───────────────────────────────────────────────────────
    {
        "id": "kandy_founded",
        "label": "Kingdom of Kandy established",
        "year_from": 1469, "year_to": 1469,
        "description": "The Kingdom of Kandy was founded in the central highlands of Sri Lanka.",
        "characters_aware": ["king", "citizen"],
        "tags": ["kandy", "kingdom"],
    },
    {
        "id": "kandy_geography",
        "label": "Kandy's highland geography as natural defence",
        "year_from": 1469, "year_to": 1815,
        "description": "Kandy's location in steep central highlands with dense jungle made it nearly impossible to attack with conventional armies.",
        "characters_aware": ["king", "citizen"],
        "tags": ["kandy", "defence", "geography"],
    },
    {
        "id": "portuguese_fail_kandy",
        "label": "Portuguese fail to conquer Kandy",
        "year_from": 1521, "year_to": 1638,
        "description": "Multiple Portuguese military expeditions into the Kandyan highlands failed due to terrain, guerrilla tactics, and supply line problems.",
        "characters_aware": ["king", "citizen"],
        "tags": ["kandy", "portuguese", "colonial", "misconception_target"],
    },
    {
        "id": "kandyan_dutch_alliance",
        "label": "Kandy allies with Dutch against Portuguese",
        "year_from": 1638, "year_to": 1638,
        "description": "King Rajasinha II signed a treaty with the Dutch VOC to jointly expel the Portuguese from Ceylon's coasts.",
        "characters_aware": ["king", "dutch", "citizen"],
        "tags": ["kandy", "dutch", "alliance", "colonial"],
    },
    {
        "id": "dutch_expel_portuguese",
        "label": "Dutch expel Portuguese from coastal Ceylon",
        "year_from": 1638, "year_to": 1658,
        "description": "With Kandyan support, the Dutch VOC expelled the Portuguese from all coastal regions of Ceylon by 1658.",
        "characters_aware": ["dutch", "citizen"],
        "tags": ["dutch", "portuguese", "colonial", "coast"],
    },
    {
        "id": "dutch_control_coasts",
        "label": "Dutch control Ceylon's coastal regions",
        "year_from": 1658, "year_to": 1796,
        "description": "The VOC controlled coastal Ceylon from 1658–1796 but never succeeded in conquering the Kandyan interior.",
        "characters_aware": ["dutch", "king", "citizen"],
        "tags": ["dutch", "colonial", "coast"],
    },
    {
        "id": "dutch_fail_kandy",
        "label": "Dutch fail to conquer Kandy",
        "year_from": 1658, "year_to": 1796,
        "description": "Despite controlling the coasts, the Dutch never conquered the Kingdom of Kandy.",
        "characters_aware": ["dutch", "king", "citizen"],
        "tags": ["kandy", "dutch", "colonial", "misconception_target"],
    },
    {
        "id": "king_vijaya_reign",
        "label": "King Sri Vijaya Rajasinha reigns",
        "year_from": 1739, "year_to": 1747,
        "description": "Sri Vijaya Rajasinha, of Nayak origin from South India, ruled as the King of Kandy from 1739 to 1747.",
        "characters_aware": ["king", "citizen"],
        "tags": ["king", "kandy", "reign"],
    },
    {
        "id": "british_arrive",
        "label": "British arrive in Ceylon",
        "year_from": 1796, "year_to": 1796,
        "description": "Britain occupied Dutch Ceylon in 1796 during the Napoleonic Wars when the Netherlands fell under French control.",
        "characters_aware": ["citizen"],
        "tags": ["british", "colonial"],
    },
    {
        "id": "kandyan_convention",
        "label": "Kandyan Convention — Kandy cedes to British",
        "year_from": 1815, "year_to": 1815,
        "description": "Kandy ceded to the British through the Kandyan Convention of 1815, not through military conquest. Internal political divisions were the primary cause.",
        "characters_aware": ["citizen"],
        "tags": ["kandy", "british", "colonial", "landmark", "misconception_target"],
    },
    {
        "id": "british_protect_relic",
        "label": "British promise to protect Buddhist institutions",
        "year_from": 1815, "year_to": 1815,
        "description": "The Kandyan Convention included British guarantees to protect Buddhist institutions and the Temple of the Tooth.",
        "characters_aware": ["citizen"],
        "tags": ["british", "buddhism", "relic", "kandy"],
    },
    {
        "id": "independence",
        "label": "Ceylon gains independence",
        "year_from": 1948, "year_to": 1948,
        "description": "Ceylon gained independence from Britain on 4 February 1948.",
        "characters_aware": ["citizen"],
        "tags": ["independence", "modern"],
    },
    {
        "id": "republic_1972",
        "label": "Ceylon becomes Republic of Sri Lanka",
        "year_from": 1972, "year_to": 1972,
        "description": "Ceylon was renamed the Republic of Sri Lanka in 1972, not in 1948.",
        "characters_aware": ["citizen"],
        "tags": ["independence", "modern", "misconception_target"],
    },

    # ── Dutch Trade ───────────────────────────────────────────────────────────
    {
        "id": "portuguese_original_galle",
        "label": "Portuguese build original Galle fortification",
        "year_from": 1588, "year_to": 1588,
        "description": "The Portuguese built a small fort at Galle in 1588 — the original structure before Dutch expansion.",
        "characters_aware": ["dutch", "citizen"],
        "tags": ["galle", "portuguese", "fort", "misconception_target"],
    },
    {
        "id": "dutch_expand_galle",
        "label": "Dutch massively expand Galle Fort",
        "year_from": 1663, "year_to": 1669,
        "description": "The Dutch rebuilt and massively expanded the fort between 1663–1669, creating 14 bastions across 36 hectares.",
        "characters_aware": ["dutch", "citizen"],
        "tags": ["galle", "fort", "dutch", "landmark"],
    },
    {
        "id": "voc_cinnamon_monopoly",
        "label": "VOC establishes cinnamon monopoly",
        "year_from": 1658, "year_to": 1796,
        "description": "The Dutch East India Company (VOC) controlled the global cinnamon trade through strict monopoly over Ceylon production.",
        "characters_aware": ["dutch", "citizen"],
        "tags": ["trade", "cinnamon", "dutch", "voc"],
    },
    {
        "id": "cinnamon_salagama",
        "label": "Salagama caste compelled to peel cinnamon",
        "year_from": 1658, "year_to": 1796,
        "description": "The Dutch compelled the Salagama caste to peel cinnamon under their rule, reshaping social structures.",
        "characters_aware": ["dutch", "citizen"],
        "tags": ["trade", "cinnamon", "dutch", "social"],
    },
    {
        "id": "british_inherit_cinnamon",
        "label": "British inherit Dutch cinnamon monopoly",
        "year_from": 1796, "year_to": 1833,
        "description": "When the British took over, they initially maintained the cinnamon monopoly before eventually opening free trade.",
        "characters_aware": ["citizen"],
        "tags": ["british", "trade", "cinnamon"],
    },

    # ── Buddhism in Lanka (early) ─────────────────────────────────────────────
    {
        "id": "buddhism_arrives_lanka",
        "label": "Buddhism arrives in Sri Lanka",
        "year_from": -247, "year_to": -247,
        "description": "Buddhism arrived in Sri Lanka in the 3rd century BCE via Mahinda, son of Emperor Ashoka, during the reign of King Devanampiya Tissa.",
        "characters_aware": ["citizen"],
        "tags": ["buddhism", "ancient", "landmark"],
    },
    {
        "id": "pre_buddhist_religion",
        "label": "Pre-Buddhist traditions in Sri Lanka",
        "year_from": -1000, "year_to": -247,
        "description": "Before Buddhism, animist and early Hindu traditions were practised in Sri Lanka.",
        "characters_aware": ["citizen"],
        "tags": ["religion", "ancient", "misconception_target"],
    },
]

GRAPH_EDGES = [
    # Relic chain
    ("buddha_death",           "relic_india",              "CAUSES",   "The Buddha's death produced the relics"),
    ("relic_india",            "relic_arrives_lanka",      "PRECEDES", "Centuries of Indian custody before Lanka arrival"),
    ("relic_arrives_lanka",    "relic_legitimises_rule",   "CAUSES",   "Possession of Relic → political legitimacy"),
    ("relic_legitimises_rule", "dalada_maligawa_built",    "ENABLES",  "Need to house Relic led to Maligawa construction"),
    ("relic_arrives_lanka",    "esala_perahera_origin",    "ENABLES",  "Arrival of Relic strengthened festival traditions"),
    ("dalada_maligawa_built",  "perahera_replica_not_real","CAUSES",   "Maligawa custody → replica used in procession for safety"),
    ("relic_legitimises_rule", "kandy_geography",          "ENABLES",  "Protecting the Relic justified defending Kandy"),
    ("relic_legitimises_rule", "british_protect_relic",    "ENABLES",  "Relic's political importance required British guarantees"),

    # Kandy independence chain
    ("kandy_founded",          "kandy_geography",          "ENABLES",  "Founders chose highland location for defence"),
    ("kandy_geography",        "portuguese_fail_kandy",    "CAUSES",   "Geography defeated Portuguese armies"),
    ("portuguese_fail_kandy",  "kandyan_dutch_alliance",   "ENABLES",  "Continued Portuguese threat pushed Kandy to ally Dutch"),
    ("kandyan_dutch_alliance", "dutch_expel_portuguese",   "CAUSES",   "Joint action expelled Portuguese"),
    ("dutch_expel_portuguese", "dutch_control_coasts",     "CAUSES",   "Filling the vacuum after Portuguese"),
    ("dutch_control_coasts",   "dutch_fail_kandy",         "PRECEDES", "Dutch held coasts but never took interior"),
    ("kandy_geography",        "dutch_fail_kandy",         "CAUSES",   "Same geography that stopped Portuguese stopped Dutch"),
    ("dutch_fail_kandy",       "king_vijaya_reign",        "PRECEDES", "Vijaya reigned while Dutch held coasts but not Kandy"),
    ("king_vijaya_reign",      "british_arrive",            "PRECEDES", "After Vijaya's reign, Dutch control continued until British arrived"),
    ("british_arrive",         "kandyan_convention",       "PRECEDES", "British presence led eventually to Convention"),
    ("kandyan_convention",     "british_protect_relic",    "CAUSES",   "Convention included Buddhist protection clause"),
    ("independence",           "republic_1972",            "PRECEDES", "Independence preceded republican status by 24 years"),

    # Dutch trade chain
    ("portuguese_original_galle", "dutch_expand_galle",   "PRECEDES", "Portuguese built first; Dutch massively expanded"),
    ("dutch_expel_portuguese",    "voc_cinnamon_monopoly","CAUSES",   "Coastal control enabled cinnamon monopoly"),
    ("voc_cinnamon_monopoly",     "cinnamon_salagama",    "CAUSES",   "Monopoly enforcement required controlled labour"),
    ("british_arrive",            "british_inherit_cinnamon","CAUSES","British takeover transferred monopoly"),
    ("voc_cinnamon_monopoly",     "british_inherit_cinnamon","PRECEDES","Dutch monopoly → British monopoly"),

    # Buddhism chain
    ("buddhism_arrives_lanka",    "dalada_maligawa_built", "ENABLES",  "Buddhist traditions required sacred housing for Relic"),
    ("pre_buddhist_religion",     "buddhism_arrives_lanka","PRECEDES", "Pre-Buddhist era preceded Buddhist arrival"),
]

# ─────────────────────────────────────────────────────────────────────────────
#  MISCONCEPTION DATABASE  (used by ContradictionEngine)
# ─────────────────────────────────────────────────────────────────────────────

MISCONCEPTIONS = [
    {
        "id": "kandy_conquered_portuguese",
        "triggers": [
            "portuguese conquered kandy", "portuguese took kandy",
            "portuguese captured kandy", "portuguese won kandy",
            "portuguese controlled kandy",
        ],
        "false_claim_node": "portuguese_fail_kandy",
        "correct_node": "portuguese_fail_kandy",
        "correction": (
            "Kandy was NEVER conquered by the Portuguese. "
            "The Portuguese made multiple costly failed expeditions into the Kandyan highlands "
            "between 1521 and 1638. The terrain, guerrilla tactics, and supply line problems "
            "defeated every Portuguese army. Kandy remained sovereign."
        ),
        "severity": "high",
        "causal_path": ["kandy_geography", "portuguese_fail_kandy"],
    },
    {
        "id": "kandy_conquered_dutch",
        "triggers": [
            "dutch conquered kandy", "dutch took kandy",
            "dutch captured kandy", "dutch controlled kandy interior",
            "europeans conquered kandy",
        ],
        "false_claim_node": "dutch_fail_kandy",
        "correct_node": "dutch_fail_kandy",
        "correction": (
            "The Dutch also never conquered Kandy. While the VOC controlled the coastal regions "
            "from 1658–1796, they — like the Portuguese before them — could never penetrate the "
            "central highlands. Kandy remained the only independent kingdom in Sri Lanka until 1815."
        ),
        "severity": "high",
        "causal_path": ["kandy_geography", "dutch_fail_kandy"],
    },
    {
        "id": "actual_relic_in_perahera",
        "triggers": [
            "actual tooth relic in perahera", "real tooth carried in procession",
            "tooth relic walks through", "tooth relic paraded",
            "actual relic paraded", "relic goes through street",
            "actual tooth relic is carried", "tooth relic is carried",
            "relic is carried in", "relic carried in perahera",
        ],
        "false_claim_node": "perahera_replica_not_real",
        "correct_node": "perahera_replica_not_real",
        "correction": (
            "The actual Sacred Tooth Relic does NOT leave the temple during the Esala Perahera. "
            "What the lead elephant carries is an ornate golden REPLICA reliquary casket. "
            "The actual Tooth Relic remains secured in the inner shrine of Sri Dalada Maligawa "
            "throughout the entire festival."
        ),
        "severity": "medium",
        "causal_path": ["dalada_maligawa_built", "perahera_replica_not_real"],
    },
    {
        "id": "dutch_built_galle_scratch",
        "triggers": [
            "dutch built galle fort from scratch", "dutch created galle fort",
            "dutch invented galle fort", "dutch designed galle fort",
            "voc built galle fort", "galle fort is dutch",
        ],
        "false_claim_node": "portuguese_original_galle",
        "correct_node": "dutch_expand_galle",
        "correction": (
            "Galle Fort was NOT built from scratch by the Dutch. "
            "The Portuguese constructed the original small fortification at Galle in 1588. "
            "The Dutch then massively expanded and rebuilt it between 1663–1669, creating the "
            "14-bastion structure we see today. It has Portuguese origins and Dutch expansion."
        ),
        "severity": "medium",
        "causal_path": ["portuguese_original_galle", "dutch_expand_galle"],
    },
    {
        "id": "kandy_military_conquest_1815",
        "triggers": [
            "british conquered kandy", "british military conquest kandy",
            "british defeated kandy army", "british invaded kandy",
            "kandy fell to british army",
        ],
        "false_claim_node": "kandyan_convention",
        "correct_node": "kandyan_convention",
        "correction": (
            "Kandy did NOT fall to British military conquest. "
            "The Kandyan Convention of 1815 was a diplomatic agreement, driven primarily by internal "
            "political divisions among Kandyan nobles rather than British military victory. "
            "It was the first time in 300 years of colonial pressure that Kandy ceded sovereignty — "
            "and it was through politics, not battlefield defeat."
        ),
        "severity": "high",
        "causal_path": ["british_arrive", "kandyan_convention"],
    },
    {
        "id": "sri_lanka_always_buddhist",
        "triggers": [
            "always been buddhist", "always buddhist", "buddhism always in sri lanka",
            "sri lanka was always buddhist", "never had other religions",
        ],
        "false_claim_node": "pre_buddhist_religion",
        "correct_node": "buddhism_arrives_lanka",
        "correction": (
            "Sri Lanka was not always Buddhist. Before Buddhism arrived in the 3rd century BCE, "
            "animist and early Hindu traditions were practised on the island. "
            "Even today, Sri Lanka has significant Hindu, Muslim, and Christian communities. "
            "Buddhism arrived with Mahinda, son of Emperor Ashoka, around 247 BCE."
        ),
        "severity": "low",
        "causal_path": ["pre_buddhist_religion", "buddhism_arrives_lanka"],
    },
    {
        "id": "republic_at_independence",
        "triggers": [
            "became republic in 1948", "republic 1948", "1948 republic",
            "independent republic 1948", "sri lanka republic independence",
            "sri lanka became a republic in 1948", "republic at independence",
            "became republic when independent",
        ],
        "false_claim_node": "republic_1972",
        "correct_node": "republic_1972",
        "correction": (
            "Sri Lanka did NOT become a republic at independence. "
            "Ceylon gained independence on 4 February 1948 but remained a Dominion "
            "within the Commonwealth with the British monarch as head of state. "
            "It became the Republic of Sri Lanka only in 1972 — 24 years after independence."
        ),
        "severity": "medium",
        "causal_path": ["independence", "republic_1972"],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
#  CHARACTER TEMPORAL WINDOWS  — what each character can possibly know
# ─────────────────────────────────────────────────────────────────────────────

CHARACTER_WINDOWS = {
    "king":    {"year_from": 1600, "year_to": 1747, "name": "King Sri Vijaya Rajasinha"},
    "nilame":  {"year_from": 1400, "year_to": 1900, "name": "Diyawadana Nilame"},
    "dutch":   {"year_from": 1600, "year_to": 1800, "name": "Captain Willem van der Berg"},
    "citizen": {"year_from": -500, "year_to": 2030, "name": "Sri Lankan Historian"},
}

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN GRAPH ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class CausalKnowledgeGraph:
    """
    NetworkX-based directed causal knowledge graph.
    Nodes = historical facts/events.
    Edges = causal/temporal relationships.
    """

    def __init__(self):
        if not NX_AVAILABLE:
            raise RuntimeError("networkx not installed. Run: pip install networkx")

        self.G: nx.DiGraph = nx.DiGraph()
        self._node_map: Dict[str, dict] = {}
        self._build()
        print(f"[CausalKnowledgeGraph] Built: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")

    def _build(self):
        """Populate graph from static data above."""
        for n in GRAPH_NODES:
            self.G.add_node(
                n["id"],
                label=n["label"],
                year_from=n["year_from"],
                year_to=n["year_to"],
                description=n["description"],
                characters_aware=n.get("characters_aware", []),
                tags=n.get("tags", []),
            )
            self._node_map[n["id"]] = n

        for src, dst, rel, reason in GRAPH_EDGES:
            if self.G.has_node(src) and self.G.has_node(dst):
                self.G.add_edge(src, dst, relation=rel, reason=reason)

    # ── Node access ───────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[dict]:
        if self.G.has_node(node_id):
            return dict(self.G.nodes[node_id])
        return None

    def nodes_by_tag(self, tag: str) -> List[dict]:
        return [
            {"id": nid, **dict(data)}
            for nid, data in self.G.nodes(data=True)
            if tag in data.get("tags", [])
        ]

    # ── Causal path finding ───────────────────────────────────────────────────

    def get_causal_path(
        self, source_id: str, target_id: str
    ) -> Optional[Dict]:
        """
        Find the shortest causal path from source to target.
        Returns structured chain with event details and edge reasons.
        """
        if not (self.G.has_node(source_id) and self.G.has_node(target_id)):
            return None
        try:
            path = nx.shortest_path(self.G, source=source_id, target=target_id)
        except nx.NetworkXNoPath:
            return None

        steps = []
        for i, nid in enumerate(path):
            node = dict(self.G.nodes[nid])
            step = {
                "step": i + 1,
                "event_id": nid,
                "event": node["label"],
                "year": f"{node['year_from']}" if node["year_from"] == node["year_to"]
                        else f"{node['year_from']}–{node['year_to']}",
                "description": node["description"],
            }
            if i < len(path) - 1:
                edge = self.G.edges[nid, path[i + 1]]
                step["leads_to"] = edge.get("reason", "")
                step["relation"] = edge.get("relation", "CAUSES")
            steps.append(step)

        src_node = dict(self.G.nodes[source_id])
        tgt_node = dict(self.G.nodes[target_id])
        return {
            "title": f"From '{src_node['label']}' to '{tgt_node['label']}'",
            "source": source_id,
            "target": target_id,
            "steps": steps,
            "total_steps": len(steps),
            "year_span": f"{src_node['year_from']} → {tgt_node['year_to']}",
        }

    def get_all_paths(
        self, source_id: str, target_id: str, max_paths: int = 3
    ) -> List[Dict]:
        """Find all simple paths from source to target (up to max_paths)."""
        if not (self.G.has_node(source_id) and self.G.has_node(target_id)):
            return []
        try:
            all_paths = list(nx.all_simple_paths(self.G, source_id, target_id))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
        all_paths.sort(key=len)
        results = []
        for path in all_paths[:max_paths]:
            steps = []
            for i, nid in enumerate(path):
                node = dict(self.G.nodes[nid])
                step = {
                    "step": i + 1,
                    "event_id": nid,
                    "event": node["label"],
                    "year": f"{node['year_from']}",
                    "description": node["description"],
                }
                if i < len(path) - 1:
                    edge = self.G.edges[nid, path[i + 1]]
                    step["leads_to"] = edge.get("reason", "")
                steps.append(step)
            results.append({"path_length": len(path), "steps": steps})
        return results

    # ── Keyword-based chain lookup ─────────────────────────────────────────────

    QUERY_CHAINS = {
        "kandyan_independence": {
            "keywords": [
                "kandy remain independent", "why kandy", "kandy survived",
                "kandy resist", "how kandy", "kandy colonial", "kandy independence",
                "why didn't", "why did kandy",
            ],
            "source": "kandy_founded",
            "target": "kandyan_convention",
        },
        "dutch_cinnamon": {
            "keywords": [
                "cinnamon trade", "dutch trade", "voc trade", "spice trade",
                "cinnamon monopoly", "dutch economic", "voc economic",
            ],
            "source": "dutch_expel_portuguese",
            "target": "british_inherit_cinnamon",
        },
        "tooth_relic_power": {
            "keywords": [
                "why tooth relic", "relic power", "relic king", "relic legitimacy",
                "why relic important", "significance relic",
            ],
            "source": "relic_arrives_lanka",
            "target": "british_protect_relic",
        },
        "galle_fort_history": {
            "keywords": [
                "galle fort history", "who built galle", "galle history",
                "history of galle", "galle fort built",
            ],
            "source": "portuguese_original_galle",
            "target": "dutch_expand_galle",
        },
        "buddhism_history": {
            "keywords": [
                "buddhism arrive", "when buddhism", "how buddhism",
                "buddhism history", "history of buddhism lanka",
            ],
            "source": "pre_buddhist_religion",
            "target": "dalada_maligawa_built",
        },
        "relic_journey": {
            "keywords": [
                "relic journey", "how relic came", "relic origin",
                "where relic came from", "relic history",
            ],
            "source": "buddha_death",
            "target": "dalada_maligawa_built",
        },
    }

    def find_chain_for_query(self, query: str) -> Optional[Dict]:
        q = query.lower()
        for chain_key, chain_data in self.QUERY_CHAINS.items():
            if any(kw in q for kw in chain_data["keywords"]):
                return self.get_causal_path(
                    chain_data["source"], chain_data["target"]
                )
        # Fallback: try to match node labels
        for nid, data in self.G.nodes(data=True):
            if any(tag in q for tag in data.get("tags", [])):
                # Return the upstream causes of the first match
                ancestors = list(nx.ancestors(self.G, nid))
                if ancestors:
                    # Find the most distant ancestor (root)
                    root = min(
                        ancestors,
                        key=lambda a: self.G.nodes[a].get("year_from", 9999)
                    )
                    return self.get_causal_path(root, nid)
        return None

    def format_chain_text(self, chain: Dict) -> str:
        """Format a chain into human-readable text for injection into LLM prompt."""
        if not chain:
            return ""
        lines = [f"CAUSAL CHAIN: {chain['title']} ({chain['year_span']})", ""]
        for step in chain["steps"]:
            lines.append(
                f"Step {step['step']}: {step['event']} ({step['year']})"
            )
            lines.append(f"  → {step['description']}")
            if step.get("leads_to"):
                lines.append(f"  BECAUSE: {step['leads_to']}")
            lines.append("")
        return "\n".join(lines)

    # ── Temporal constraint check ──────────────────────────────────────────────

    def check_anachronism(
        self, claim_node_id: str, character_id: str
    ) -> Dict:
        """
        Verify whether a character could plausibly know about a fact
        given their temporal window.
        """
        window = CHARACTER_WINDOWS.get(character_id)
        if not window:
            return {"anachronism": False, "reason": "Unknown character"}

        node = self.get_node(claim_node_id)
        if not node:
            return {"anachronism": False, "reason": "Unknown node"}

        # The character's knowledge window
        char_from = window["year_from"]
        char_to   = window["year_to"]
        fact_from = node["year_from"]

        # Anachronism: fact happened AFTER character's knowledge window
        if fact_from > char_to:
            return {
                "anachronism": True,
                "severity": "high",
                "character": window["name"],
                "fact": node["label"],
                "fact_year": fact_from,
                "character_window_end": char_to,
                "correction": (
                    f"{window['name']} (knowledge window: {char_from}–{char_to}) "
                    f"could not know about '{node['label']}' which occurred in {fact_from}."
                ),
            }
        # Pre-anachronism: character is asked about something very old they'd likely reference
        return {"anachronism": False}

    # ── Graph statistics ───────────────────────────────────────────────────────

    def get_graph_stats(self) -> Dict:
        try:
            # Count edges by relation type
            relation_counts: dict = defaultdict(int)
            for _, _, data in self.G.edges(data=True):
                relation_counts[data.get("relation", "UNKNOWN")] += 1

            # Find root nodes (no incoming edges)
            roots = [n for n in self.G.nodes() if self.G.in_degree(n) == 0]
            # Find leaf nodes (no outgoing edges)
            leaves = [n for n in self.G.nodes() if self.G.out_degree(n) == 0]

            return {
                "total_nodes": self.G.number_of_nodes(),
                "total_edges": self.G.number_of_edges(),
                "relation_types": dict(relation_counts),
                "root_events": [self.G.nodes[r]["label"] for r in roots],
                "leaf_events": [self.G.nodes[l]["label"] for l in leaves],
                "is_dag": nx.is_directed_acyclic_graph(self.G),
                "longest_path_length": nx.dag_longest_path_length(self.G)
                    if nx.is_directed_acyclic_graph(self.G) else -1,
                "average_degree": round(
                    sum(dict(self.G.degree()).values()) / max(self.G.number_of_nodes(), 1), 2
                ),
            }
        except Exception as e:
            return {"error": str(e)}

    def export_graph_json(self) -> Dict:
        """Export the full graph as JSON (for Flutter visualization)."""
        nodes_out = []
        for nid, data in self.G.nodes(data=True):
            nodes_out.append({
                "id": nid,
                "label": data.get("label", nid),
                "year_from": data.get("year_from"),
                "year_to": data.get("year_to"),
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "characters_aware": data.get("characters_aware", []),
            })

        edges_out = []
        for src, dst, data in self.G.edges(data=True):
            edges_out.append({
                "source": src,
                "target": dst,
                "relation": data.get("relation", "CAUSES"),
                "reason": data.get("reason", ""),
            })

        return {
            "nodes": nodes_out,
            "edges": edges_out,
            "stats": self.get_graph_stats(),
            "generated_at": datetime.now().isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  CONTRADICTION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ContradictionEngine:
    """
    Validates historical claims against the knowledge graph.
    Detects:
      1. Known misconceptions (trigger-phrase matching)
      2. Temporal anachronisms (wrong century for a character)
      3. Reversed causality (claiming B caused A when graph shows A→B)
      4. False premises in questions
    """

    def __init__(self, graph: CausalKnowledgeGraph):
        self.graph = graph
        self._misconceptions = MISCONCEPTIONS
        print("[ContradictionEngine] Initialized with "
              f"{len(self._misconceptions)} misconception rules")

    # ── Misconception detection ────────────────────────────────────────────────

    def detect_misconception(self, query: str) -> Optional[Dict]:
        q = query.lower()
        for m in self._misconceptions:
            if any(trigger in q for trigger in m["triggers"]):
                path = None
                if len(m.get("causal_path", [])) >= 2:
                    path = self.graph.get_causal_path(
                        m["causal_path"][0], m["causal_path"][-1]
                    )
                return {
                    "misconception_id": m["id"],
                    "severity": m["severity"],
                    "correction": m["correction"],
                    "supporting_chain": path,
                    "false_node_id": m.get("false_claim_node"),
                    "correct_node_id": m.get("correct_node"),
                }
        return None

    # ── Reversed causality detection ──────────────────────────────────────────

    def detect_reversed_causality(self, query: str) -> Optional[Dict]:
        """
        Detect patterns like 'X caused Y' when the graph shows Y→X.
        Simple heuristic: look for CAUSE/LED patterns and check direction.
        """
        q = query.lower()
        # Patterns: "X caused Y", "Y was because of X", "X led to Y"
        patterns = [
            r"(\w[\w ]+) caused (\w[\w ]+)",
            r"(\w[\w ]+) led to (\w[\w ]+)",
            r"(\w[\w ]+) resulted in (\w[\w ]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            claimed_cause = match.group(1).strip()
            claimed_effect = match.group(2).strip()
            # Try to find nodes that match these labels
            cause_node = self._fuzzy_find_node(claimed_cause)
            effect_node = self._fuzzy_find_node(claimed_effect)
            if cause_node and effect_node:
                # Check if the graph has the edge in the OPPOSITE direction
                if self.graph.G.has_edge(effect_node, cause_node):
                    src_label = self.graph.G.nodes[effect_node]["label"]
                    tgt_label = self.graph.G.nodes[cause_node]["label"]
                    edge = self.graph.G.edges[effect_node, cause_node]
                    return {
                        "reversed_causality": True,
                        "claimed_cause": claimed_cause,
                        "claimed_effect": claimed_effect,
                        "actual_direction": f"'{src_label}' → '{tgt_label}'",
                        "edge_reason": edge.get("reason", ""),
                        "correction": (
                            f"The causal direction is actually reversed. "
                            f"In the historical record, '{src_label}' "
                            f"{edge.get('relation','CAUSED').lower()} "
                            f"'{tgt_label}', not the other way around."
                        ),
                    }
        return None

    def _fuzzy_find_node(self, text: str) -> Optional[str]:
        """Find a graph node whose tags or label words appear in the text."""
        text_words = set(text.lower().split())
        best_nid = None
        best_score = 0
        for nid, data in self.graph.G.nodes(data=True):
            tags = set(data.get("tags", []))
            label_words = set(data.get("label", "").lower().split())
            score = len(text_words & (tags | label_words))
            if score > best_score:
                best_score = score
                best_nid = nid
        return best_nid if best_score >= 1 else None

    # ── Anachronism detection ─────────────────────────────────────────────────

    def detect_anachronism(self, query: str, character_id: str) -> Optional[Dict]:
        """
        Check if the query contains references to events the character
        cannot know about given their temporal window.
        """
        window = CHARACTER_WINDOWS.get(character_id)
        if not window:
            return None

        q = query.lower()
        future_nodes = []
        for nid, data in self.graph.G.nodes(data=True):
            # Only check events after the character's window
            if data.get("year_from", 0) > window["year_to"]:
                label_words = data.get("label", "").lower().split()
                tags = data.get("tags", [])
                if any(w in q for w in label_words + tags):
                    future_nodes.append({
                        "node_id": nid,
                        "label": data["label"],
                        "year": data["year_from"],
                    })

        if future_nodes:
            earliest = min(future_nodes, key=lambda x: x["year"])
            return {
                "anachronism_detected": True,
                "character": window["name"],
                "character_window_end": window["year_to"],
                "future_reference": earliest["label"],
                "future_year": earliest["year"],
                "correction": (
                    f"As {window['name']} (living until ~{window['year_to']}), "
                    f"I cannot speak to '{earliest['label']}' which occurred in {earliest['year']}. "
                    f"Please ask our modern historian (the Citizen character) for that."
                ),
                "all_future_refs": future_nodes,
            }
        return None

    # ── Full validate pipeline ─────────────────────────────────────────────────

    def validate_claim(
        self, query: str, character_id: str = "citizen"
    ) -> Dict:
        """
        Run all checks and return a structured validation result.
        Always includes the relevant causal chain if found.
        """
        result: Dict = {
            "query": query,
            "character_id": character_id,
            "misconception": None,
            "reversed_causality": None,
            "anachronism": None,
            "causal_chain": None,
            "has_contradiction": False,
            "severity": "none",
            "combined_correction": None,
        }

        # 1. Misconception check
        misconception = self.detect_misconception(query)
        if misconception:
            result["misconception"] = misconception
            result["has_contradiction"] = True
            result["severity"] = misconception["severity"]
            result["causal_chain"] = misconception.get("supporting_chain")
            result["combined_correction"] = (
                "⚠ HISTORICAL CORRECTION:\n" + misconception["correction"]
            )

        # 2. Reversed causality
        if not result["has_contradiction"]:
            rev = self.detect_reversed_causality(query)
            if rev:
                result["reversed_causality"] = rev
                result["has_contradiction"] = True
                result["severity"] = "medium"
                result["combined_correction"] = (
                    "⚠ CAUSAL DIRECTION ERROR:\n" + rev["correction"]
                )

        # 3. Anachronism (only for historical characters, not citizen)
        if character_id != "citizen":
            ana = self.detect_anachronism(query, character_id)
            if ana:
                result["anachronism"] = ana
                result["has_contradiction"] = result.get("has_contradiction", False)
                # Anachronisms are additive — append to correction
                ana_text = "⏳ ANACHRONISM NOTE:\n" + ana["correction"]
                if result["combined_correction"]:
                    result["combined_correction"] += "\n\n" + ana_text
                else:
                    result["combined_correction"] = ana_text

        # 4. Always find the relevant causal chain if not already set
        if not result["causal_chain"]:
            chain = self.graph.find_chain_for_query(query)
            result["causal_chain"] = chain

        return result

    # ── Post-generation response validator ────────────────────────────────────

    def validate_response(
        self, response_text: str, character_id: str
    ) -> Dict:
        """
        Validate the LLM's generated response against the graph.
        Scans the response for node labels and checks:
          - Are all mentioned events within the character's time window?
          - Does the response imply any known false causal relationships?
        Returns rewrite suggestions if issues found.
        """
        window = CHARACTER_WINDOWS.get(character_id)
        issues = []

        if window:
            for nid, data in self.graph.G.nodes(data=True):
                label_lower = data.get("label", "").lower()
                if label_lower in response_text.lower():
                    # Check if this fact is outside the character's window
                    if data.get("year_from", 0) > window["year_to"]:
                        issues.append({
                            "type": "anachronism_in_response",
                            "node": nid,
                            "fact": data["label"],
                            "fact_year": data["year_from"],
                            "character_window_end": window["year_to"],
                        })

        # Check if any known misconception text appears in the response
        for m in self._misconceptions:
            for trigger in m["triggers"][:2]:  # Only top 2 triggers to avoid false positives
                if trigger in response_text.lower():
                    issues.append({
                        "type": "misconception_in_response",
                        "misconception_id": m["id"],
                        "trigger_found": trigger,
                        "correction": m["correction"],
                    })

        return {
            "issues_found": len(issues),
            "issues": issues,
            "response_is_clean": len(issues) == 0,
            "requires_rewrite": any(
                i["type"] == "misconception_in_response" for i in issues
            ),
        }

    def get_all_misconceptions(self) -> List[Dict]:
        """Return the full misconception database (for admin/debug)."""
        return [
            {
                "id": m["id"],
                "severity": m["severity"],
                "correction": m["correction"],
                "trigger_count": len(m["triggers"]),
                "sample_trigger": m["triggers"][0] if m["triggers"] else "",
            }
            for m in self._misconceptions
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK ENDPOINT REGISTRATION  — call this from create_flask_api()
# ─────────────────────────────────────────────────────────────────────────────

def register_causal_endpoints(app, chatbot):
    """
    Register all /causal/* and /graph/* endpoints on an existing Flask app.
    Call this at the end of create_flask_api() before returning.

    Usage in inference_api.py::create_flask_api():
        from causal_knowledge_graph import register_causal_endpoints
        register_causal_endpoints(app, chatbot)
        return app
    """
    from flask import request, jsonify

    graph  = chatbot.causal_graph   # CausalKnowledgeGraph instance
    engine = chatbot.contradiction_engine  # ContradictionEngine instance

    # ── GET /causal/graph  — export full graph JSON ───────────────────────────
    @app.route("/causal/graph", methods=["GET"])
    def causal_graph_export():
        """Export the full knowledge graph as JSON for Flutter DAG visualization."""
        try:
            return jsonify({
                "success": True,
                "graph": graph.export_graph_json(),
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ── GET /causal/stats ──────────────────────────────────────────────────────
    @app.route("/causal/stats", methods=["GET"])
    def causal_stats():
        try:
            return jsonify({
                "success": True,
                "stats": graph.get_graph_stats(),
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ── POST /causal/chain  — get chain for a query ────────────────────────────
    @app.route("/causal/chain", methods=["GET", "POST"])
    def causal_chain_query():
        if request.method == "GET":
            return jsonify({
                "info": "POST a historical query to get its causal chain.",
                "example": {"query": "Why did Kandy remain independent?"},
                "available_query_topics": list(graph.QUERY_CHAINS.keys()),
            }), 200
        try:
            data  = request.get_json(force=True, silent=True) or {}
            query = data.get("query", "")
            if not query:
                return jsonify({"error": "query required"}), 400
            chain = graph.find_chain_for_query(query)
            return jsonify({
                "success": True,
                "query": query,
                "chain_found": chain is not None,
                "chain": chain,
                "formatted_text": graph.format_chain_text(chain) if chain else None,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ── POST /causal/path  — explicit source→target path ──────────────────────
    @app.route("/causal/path", methods=["POST"])
    def causal_path():
        """
        Find the causal path between two explicit node IDs.
        POST: {"source": "kandy_founded", "target": "kandyan_convention"}
        """
        try:
            data   = request.get_json(force=True, silent=True) or {}
            source = data.get("source", "")
            target = data.get("target", "")
            if not source or not target:
                return jsonify({
                    "error": "Both 'source' and 'target' node IDs required",
                    "example": {"source": "kandy_founded", "target": "kandyan_convention"},
                    "available_nodes": [n for n in graph.G.nodes()],
                }), 400
            path = graph.get_causal_path(source, target)
            all_paths = graph.get_all_paths(source, target, max_paths=3)
            return jsonify({
                "success": True,
                "source": source,
                "target": target,
                "shortest_path": path,
                "all_paths": all_paths,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ── POST /causal/validate  — full claim validation ─────────────────────────
    @app.route("/causal/validate", methods=["POST"])
    def causal_validate():
        """
        Validate a claim/query against the knowledge graph.
        Returns misconceptions, reversed causality, anachronisms, and the correct chain.
        """
        try:
            data         = request.get_json(force=True, silent=True) or {}
            query        = data.get("query", "")
            character_id = data.get("character_id", "citizen")
            if not query:
                return jsonify({"error": "query required"}), 400
            validation = engine.validate_claim(query, character_id)
            return jsonify({
                "success": True,
                **validation,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ── POST /causal/check  — misconception-only fast check ───────────────────
    @app.route("/causal/check", methods=["POST"])
    def causal_check():
        """
        Fast misconception check only (no full validation pipeline).
        Suitable for real-time client-side checks before submitting a question.
        """
        try:
            data  = request.get_json(force=True, silent=True) or {}
            query = data.get("query", "")
            if not query:
                return jsonify({"error": "query required"}), 400
            misconception = engine.detect_misconception(query)
            return jsonify({
                "success": True,
                "query": query,
                "misconception_detected": misconception is not None,
                "result": misconception,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ── POST /causal/validate-response  — post-gen response check ─────────────
    @app.route("/causal/validate-response", methods=["POST"])
    def causal_validate_response():
        """
        Validate a generated LLM response for anachronisms and misconceptions.
        POST: {"response": "...", "character_id": "king"}
        """
        try:
            data         = request.get_json(force=True, silent=True) or {}
            response     = data.get("response", "")
            character_id = data.get("character_id", "citizen")
            if not response:
                return jsonify({"error": "response text required"}), 400
            result = engine.validate_response(response, character_id)
            return jsonify({
                "success": True,
                "character_id": character_id,
                **result,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # ── GET /causal/misconceptions  — list all rules ───────────────────────────
    @app.route("/causal/misconceptions", methods=["GET"])
    def causal_misconceptions():
        return jsonify({
            "success": True,
            "misconceptions": engine.get_all_misconceptions(),
            "total": len(engine.get_all_misconceptions()),
            "timestamp": datetime.now().isoformat(),
        })

    # ── GET /causal/node/<node_id>  — inspect a single node ───────────────────
    @app.route("/causal/node/<node_id>", methods=["GET"])
    def causal_node(node_id):
        node = graph.get_node(node_id)
        if not node:
            return jsonify({
                "error": f"Node '{node_id}' not found",
                "available_nodes": list(graph.G.nodes()),
            }), 404
        # Include immediate causes and effects
        causes  = [
            {"node_id": p, "label": graph.G.nodes[p]["label"],
             "reason": graph.G.edges[p, node_id].get("reason", "")}
            for p in graph.G.predecessors(node_id)
        ]
        effects = [
            {"node_id": s, "label": graph.G.nodes[s]["label"],
             "reason": graph.G.edges[node_id, s].get("reason", "")}
            for s in graph.G.successors(node_id)
        ]
        return jsonify({
            "success": True,
            "node": {"id": node_id, **node},
            "direct_causes": causes,
            "direct_effects": effects,
            "timestamp": datetime.now().isoformat(),
        })

    # ── GET /causal/nodes  — list all nodes with optional tag filter ────────────
    @app.route("/causal/nodes", methods=["GET"])
    def causal_nodes():
        tag = request.args.get("tag", "")
        if tag:
            nodes = graph.nodes_by_tag(tag)
        else:
            nodes = [
                {"id": nid, **dict(data)}
                for nid, data in graph.G.nodes(data=True)
            ]
        return jsonify({
            "success": True,
            "tag_filter": tag or None,
            "nodes": nodes,
            "total": len(nodes),
            "timestamp": datetime.now().isoformat(),
        })

    print("[CausalKnowledgeGraph] Flask endpoints registered: "
          "/causal/graph  /causal/stats  /causal/chain  /causal/path  "
          "/causal/validate  /causal/check  /causal/validate-response  "
          "/causal/misconceptions  /causal/node/<id>  /causal/nodes")


# ─────────────────────────────────────────────────────────────────────────────
#  CHATBOT PIPELINE INTEGRATION  — inject into generate_answer()
# ─────────────────────────────────────────────────────────────────────────────

def integrate_with_chatbot(chatbot_instance):
    """
    Attach CausalKnowledgeGraph + ContradictionEngine to an existing
    MultiCharacterChatbot instance and patch generate_answer().

    Call once after creating the chatbot:
        chatbot = MultiCharacterChatbot(CONFIG, loader)
        integrate_with_chatbot(chatbot)
    """
    graph  = CausalKnowledgeGraph()
    engine = ContradictionEngine(graph)

    chatbot_instance.causal_graph         = graph
    chatbot_instance.contradiction_engine = engine

    # Save reference to the original method
    original_generate = chatbot_instance.generate_answer.__func__

    def patched_generate_answer(self, query, char_id,
                                session_id="default", expertise_level=None):
        # 1. Run original pipeline
        result = original_generate(
            self, query, char_id, session_id, expertise_level
        )
        if result.get("error"):
            return result

        # 2. Validate the query + generated answer against the graph
        try:
            validation = engine.validate_claim(query, char_id)

            # 3. If contradiction detected — prepend correction to answer
            if validation["has_contradiction"] and validation.get("combined_correction"):
                result["answer"] = (
                    validation["combined_correction"]
                    + "\n\n---\n\n"
                    + result["answer"]
                )

            # 4. Attach causal chain to result
            chain = validation.get("causal_chain") or \
                    graph.find_chain_for_query(query)
            if chain:
                result["causal_chain"] = graph.format_chain_text(chain)
                result["causal_chain_data"] = chain

            # 5. Attach validation summary
            result["graph_validation"] = {
                "has_contradiction":   validation["has_contradiction"],
                "severity":            validation["severity"],
                "misconception_id":    validation["misconception"]["misconception_id"]
                                       if validation["misconception"] else None,
                "anachronism":         validation["anachronism"] is not None,
                "reversed_causality":  validation["reversed_causality"] is not None,
            }
        except Exception as ex:
            print(f"[CausalEngine] Validation error (non-fatal): {ex}")

        return result

    # Bind the patched method
    import types
    chatbot_instance.generate_answer = types.MethodType(
        patched_generate_answer, chatbot_instance
    )

    print("[CausalKnowledgeGraph] Patched chatbot.generate_answer() with "
          "contradiction + causal chain injection")
    return graph, engine


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CAUSAL KNOWLEDGE GRAPH  —  Standalone Test")
    print("=" * 60)

    g  = CausalKnowledgeGraph()
    e  = ContradictionEngine(g)

    print("\n── Graph Stats ──")
    stats = g.get_graph_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n── Causal Chain: Kandyan Independence ──")
    chain = g.get_causal_path("kandy_founded", "kandyan_convention")
    if chain:
        print(g.format_chain_text(chain))

    print("\n── Causal Chain: Tooth Relic Power ──")
    chain2 = g.get_causal_path("relic_arrives_lanka", "british_protect_relic")
    if chain2:
        print(g.format_chain_text(chain2))

    print("\n── Misconception Tests ──")
    tests = [
        ("The Portuguese conquered Kandy", "king"),
        ("The actual tooth relic is carried in the Perahera", "nilame"),
        ("The Dutch built Galle Fort from scratch", "dutch"),
        ("Sri Lanka became a republic in 1948", "citizen"),
        ("British conquered Kandy militarily", "citizen"),
        ("What is the Esala Perahera?", "nilame"),  # No misconception
    ]
    for query, char in tests:
        result = e.validate_claim(query, char)
        status = "CONTRADICTION" if result["has_contradiction"] else "CLEAN"
        print(f"\n  [{status}] Q: {query[:60]}")
        if result["has_contradiction"]:
            print(f"           Severity: {result['severity']}")
            print(f"           Correction: {result['combined_correction'][:100]}...")

    print("\n── Anachronism Test (King asked about 1948) ──")
    ana = e.detect_anachronism("What happened at independence in 1948?", "king")
    if ana:
        print(f"  Anachronism: {ana['correction']}")

    print("\n── Query-based chain lookup ──")
    auto_chain = g.find_chain_for_query("Why did Kandy remain independent from colonial powers?")
    if auto_chain:
        print(f"  Found chain: {auto_chain['title']} ({auto_chain['total_steps']} steps)")

    print("\n── Export graph JSON (first 2 nodes) ──")
    exported = g.export_graph_json()
    print(f"  Total nodes: {exported['stats']['total_nodes']}")
    print(f"  Total edges: {exported['stats']['total_edges']}")
    print(f"  Is DAG:      {exported['stats']['is_dag']}")
    print(f"  Longest path: {exported['stats']['longest_path_length']} hops")
    print("\n  First node:", json.dumps(exported["nodes"][0], indent=2))