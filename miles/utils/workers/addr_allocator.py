import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PortAllocator:
    # --worker-dynamic-port-start: co-resident runs on one host need disjoint ranges, or two of
    # them can probe the same free port. None keeps each node's own service port range.
    range_start: int | None = None
    _next_port_of_ip: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def alloc(self, actor, *, node_ip: str, consecutive: int = 1, hold: bool = False) -> int:
        """Probe a free block on ``actor``'s node; ``hold`` keeps it bound in that actor until released."""
        async with self._lock:
            # The node scans from this cursor within its own service port range and wraps
            # back to the range's start; a node without a cursor yet starts there.
            # only pass range_start when set, so a node without the option keeps its own range
            range_kwargs = {} if self.range_start is None else {"range_start": self.range_start}
            # only pass hold when set: the other actors' probes do not take it
            hold_kwargs = {"hold": True} if hold else {}
            port: int = await actor._get_free_port_block.remote(
                start_port=self._next_port_of_ip.get(node_ip),
                count=consecutive,
                **range_kwargs,
                **hold_kwargs,
            )
            self._next_port_of_ip[node_ip] = port + consecutive
            return port
