import math
import random
from typing import Any, Dict, List, Optional
from pypdevs.DEVS import AtomicDEVS

from hackerthon.combat_config import damage_for_distance, hit_probability


class WorldAtomic(AtomicDEVS):
    def __init__(self, name="WorldAtomic", dt=1.0, initial_entities=None, scheduled_spawns=None):
        super().__init__(name)
        self.dt = dt
        self.time = 0.0
        self.state = "RUNNING"

        self.status_in = self.addInPort("status_in")

        self.world_out = self.addOutPort("world_out")
        self.damage_out = self.addOutPort("damage_out")
        self.spawn_out = self.addOutPort("spawn_out")

        self.entities: List[Dict[str, Any]] = initial_entities if initial_entities is not None else []
        self.pending_damages: List[Dict[str, Any]] = []
        self.pending_spawns: List[Dict[str, Any]] = []
        self.scheduled_spawns: List[Dict[str, Any]] = sorted(
            scheduled_spawns or [],
            key=lambda item: float(item.get("time", 0.0)),
        )

    def timeAdvance(self):
        if self.pending_damages or self.pending_spawns:
            return 0.0
        return self.dt

    def outputFnc(self):
        output = {}

        if self.pending_damages:
            output[self.damage_out] = list(self.pending_damages)

        if self.pending_spawns:
            output[self.spawn_out] = list(self.pending_spawns)

        if output:
            return output

        return {
            self.world_out: [
                {"time": round(self.time, 2), "entities": self.entities}
            ]
        }

    def intTransition(self):
        if self.pending_damages or self.pending_spawns:
            self.pending_damages.clear()
            self.pending_spawns.clear()
            return self.state

        self.time += self.dt
        self._release_due_spawns()
        return self.state

    def extTransition(self, inputs):
        if self.status_in in inputs:
            all_statuses = list(inputs[self.status_in])

            # Phase 1: update all entity positions and self-reported HP first.
            # Must be separate from Phase 2 so that a target's own status report
            # (hp=100, not yet damaged) cannot overwrite damage dealt in the same tick.
            for status in all_statuses:
                self._update_entity_from_status(status)

            # Phase 2: apply ENGAGE damage after all positions/HP are current.
            for status in all_statuses:
                if status.get("mode") != "ENGAGE" or status.get("target_id") is None:
                    continue

                target = next(
                    (e for e in self.entities if e.get("id") == status["target_id"]),
                    None,
                )

                if target is None:
                    continue

                if target.get("state") == "DESTROYED":
                    continue

                if target.get("hp") is not None and float(target.get("hp", 0)) <= 0:
                    continue

                distance = math.hypot(
                    float(target.get("x", 0.0)) - float(status.get("x", 0.0)),
                    float(target.get("y", 0.0)) - float(status.get("y", 0.0)),
                )
                if random.random() > hit_probability(distance):
                    continue

                damage = damage_for_distance(distance)
                if damage <= 0:
                    continue

                target["hp"] = max(0, int(target.get("hp", 100)) - damage)
                if target["hp"] <= 0:
                    target["state"] = "DESTROYED"

                self.pending_damages.append({
                    "target_id": status["target_id"],
                    "damage": damage,
                })

        return self.state

    def _release_due_spawns(self):
        remaining = []

        for spawn in self.scheduled_spawns:
            spawn_time = float(spawn.get("time", 0.0))
            if spawn_time <= self.time:
                entity = {
                    "id": spawn["id"],
                    "type": spawn.get("type", "enemy"),
                    "x": spawn.get("x", 0.0),
                    "y": spawn.get("y", 0.0),
                    "heading": spawn.get("heading", 0.0),
                    "hp": spawn.get("hp", 100),
                    "ammo": spawn.get("ammo", 30),
                    "state": "ALIVE",
                }
                self._upsert_entity(entity)
                self.pending_spawns.append({
                    "id": spawn["id"],
                    "x": spawn.get("x", 0.0),
                    "y": spawn.get("y", 0.0),
                    "heading": spawn.get("heading", 0.0),
                    "hp": spawn.get("hp", 100),
                    "ammo": spawn.get("ammo", 30),
                })
            else:
                remaining.append(spawn)

        self.scheduled_spawns = remaining

    def _upsert_entity(self, new_entity: Dict[str, Any]):
        entity_id = new_entity.get("id")
        for entity in self.entities:
            if entity.get("id") == entity_id:
                entity.update(new_entity)
                return
        self.entities.append(new_entity)

    def _update_entity_from_status(self, status: Dict[str, Any]):
        entity_id = status.get("id")
        if entity_id is None:
            return

        mode = status.get("mode", "")
        hp = status.get("hp", 100)

        for entity in self.entities:
            if entity["id"] == entity_id:
                entity["x"] = status.get("x", entity["x"])
                entity["y"] = status.get("y", entity["y"])
                entity["heading"] = status.get("heading", entity.get("heading", 0.0))
                entity["hp"] = hp

                if hp <= 0 or mode == "DESTROYED":
                    entity["state"] = "DESTROYED"
                else:
                    entity["state"] = "ALIVE"
                return

        self.entities.append({
            "id": entity_id,
            "type": "soldier",
            "x": status.get("x", 0.0),
            "y": status.get("y", 0.0),
            "heading": status.get("heading", 0.0),
            "hp": hp,
            "state": "DESTROYED" if hp <= 0 or mode == "DESTROYED" else "ALIVE",
        })
