"""Message bus abstraction.

Every stage publishes to / subscribes from a topic. Two backends:
  * InMemoryBus  – single-process queues; zero infrastructure (default).
  * KafkaBus     – real Apache Kafka via confluent-kafka (docker-compose.yml provided).

Topics used by the pipeline:
  sensor.raw, labels, windows, features, scores, events, attributions, alerts
"""
from __future__ import annotations

import json
import queue
from collections import defaultdict
from typing import Any, Callable, Iterator


class Bus:
    def publish(self, topic: str, key: str, record: dict) -> None: ...
    def subscribe(self, topic: str) -> Iterator[tuple[str, dict]]: ...
    def close(self) -> None: ...


class InMemoryBus(Bus):
    """Simple FIFO per topic. Consumers iterate with `subscribe(topic)`.

    A `None` sentinel marks end-of-stream so consumers terminate cleanly.
    """

    def __init__(self):
        self._q: dict[str, queue.Queue] = defaultdict(queue.Queue)
        self._sinks: dict[str, list[Callable[[str, dict], None]]] = defaultdict(list)

    def publish(self, topic: str, key: str, record: dict) -> None:
        self._q[topic].put((key, record))
        for fn in self._sinks[topic]:
            fn(key, record)

    def end(self, topic: str) -> None:
        self._q[topic].put(None)

    def subscribe(self, topic: str) -> Iterator[tuple[str, dict]]:
        q = self._q[topic]
        while True:
            item = q.get()
            if item is None:
                return
            yield item

    def tap(self, topic: str, fn: Callable[[str, dict], None]) -> None:
        """Attach a side listener (used by storage / evaluation)."""
        self._sinks[topic].append(fn)

    def close(self) -> None:
        pass


class KafkaBus(Bus):
    """Thin wrapper around confluent-kafka. Records are JSON-encoded."""

    def __init__(self, bootstrap: str = "localhost:9092", group: str = "rail-anomaly"):
        from confluent_kafka import Consumer, Producer  # optional dependency

        self._producer = Producer({"bootstrap.servers": bootstrap})
        self._consumer_cfg = {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "auto.offset.reset": "earliest",
        }
        self._Consumer = Consumer

    def publish(self, topic: str, key: str, record: dict) -> None:
        self._producer.produce(topic, key=key, value=json.dumps(record).encode())
        self._producer.poll(0)

    def subscribe(self, topic: str) -> Iterator[tuple[str, dict]]:
        c = self._Consumer(self._consumer_cfg)
        c.subscribe([topic])
        try:
            while True:
                msg = c.poll(1.0)
                if msg is None or msg.error():
                    continue
                yield msg.key().decode(), json.loads(msg.value())
        finally:
            c.close()

    def close(self) -> None:
        self._producer.flush()


def make_bus(kind: str = "memory", **kw: Any) -> Bus:
    return KafkaBus(**kw) if kind == "kafka" else InMemoryBus()
