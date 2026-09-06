"""Process-wide admission and cooperative compute scheduling.

Reservations are conservative working-set estimates, not allocator limits.
RSS includes native children; shared mappings may be counted more than once.
Never call checkpoint while holding a model/session/cache lock.
"""
from contextlib import contextmanager
import os
import threading
import time

from .cancellation import check_cancelled

MiB = 1024 * 1024
_local = threading.local()


def memory_state():
    import psutil
    process = psutil.Process()
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.Error:
            pass
    vm = psutil.virtual_memory()
    return rss, vm.available, vm.total


class ResourceGovernor:
    def __init__(self, limit=None, sample=memory_state, wait_seconds=30):
        self.sample = sample
        self.limit = limit
        self.wait_seconds = wait_seconds
        self.condition = threading.Condition()
        self.reservations = {}
        self.waiters = []
        self.owner = None
        self.sequence = 0
        self.last_priority = None
        self.priority_streak = 0
        self._next_memory_check = 0.0

    def _limit(self, total):
        if self.limit is not None:
            return self.limit
        try:
            requested = int(os.environ.get('RAWALCHEMY_MEMORY_LIMIT_MB', '0')) * MiB
        except ValueError:
            requested = 0
        return min(requested, total) if requested > 0 else int(total * 0.70)

    def snapshot(self):
        rss, available, total = self.sample()
        with self.condition:
            return dict(rss_bytes=rss, available_bytes=available,
                        reserved_bytes=sum(self.reservations.values()),
                        limit_bytes=self._limit(total), jobs=len(self.reservations))

    def _chosen(self):
        # Preview wins, but after three preview grants an admitted export runs.
        candidates = self.waiters
        if self.last_priority == 0 and self.priority_streak >= 3:
            background = [w for w in candidates if w[0] > 0]
            if background:
                candidates = background
        return min(candidates) if candidates else None

    def _acquire_compute(self, ticket, priority):
        entry = (priority, ticket)
        self.waiters.append(entry)
        try:
            while self.owner is not None or self._chosen() != entry:
                check_cancelled()
                self.condition.wait(0.05)
            check_cancelled()
            self.owner = ticket
            self.priority_streak = self.priority_streak + 1 if self.last_priority == priority else 1
            self.last_priority = priority
        finally:
            self.waiters.remove(entry)

    @contextmanager
    def job(self, estimate, *, priority=0):
        if getattr(_local, 'governor', None) is self:
            yield  # A nested stage retains the outer reservation.
            return
        start = time.monotonic()
        estimate = max(0, int(estimate))
        with self.condition:
            self.sequence += 1
            ticket = self.sequence
            while True:
                check_cancelled()
                rss, available, total = self.sample()
                limit = self._limit(total)
                # Existing arrays/models/Qt buffers are reflected in RSS;
                # add active estimates conservatively to reserve future growth.
                spare = min(limit - rss - sum(self.reservations.values()), available - 256 * MiB)
                if estimate <= spare:
                    self.reservations[ticket] = estimate
                    break
                if estimate > limit or time.monotonic() - start >= self.wait_seconds:
                    raise MemoryError('Image job exceeds the available memory budget; close images or reduce concurrent work')
                self.condition.wait(0.05)
            try:
                self._acquire_compute(ticket, priority)
            except BaseException:
                self.reservations.pop(ticket, None)
                self.condition.notify_all()
                raise
        previous = tuple(getattr(_local, name, None) for name in ('governor', 'ticket', 'priority'))
        _local.governor, _local.ticket, _local.priority = self, ticket, priority
        try:
            yield
        finally:
            _local.governor, _local.ticket, _local.priority = previous
            with self.condition:
                if self.owner == ticket:
                    self.owner = None
                self.reservations.pop(ticket, None)
                self.condition.notify_all()

    @contextmanager
    def maintenance(self):
        """Run global cache/session maintenance only between admitted jobs."""
        with self.condition:
            yield not self.reservations and self.owner is None

    def check_memory(self, *, force=False):
        # Native receive loops call this while owning a session lock. Never
        # yield the compute slot here: that would deadlock another stage.
        now = time.monotonic()
        if force or now >= self._next_memory_check:
            self._next_memory_check = now + 0.25
            rss, _, total = self.sample()
            if rss > self._limit(total):
                raise MemoryError('Application and native workers exceeded the memory budget')

    def checkpoint(self):
        check_cancelled()
        if getattr(_local, 'governor', None) is not self:
            return
        self.check_memory(force=True)
        with self.condition:
            ticket = _local.ticket
            if self.owner != ticket or not self.waiters:
                return
            self.owner = None
            self.condition.notify_all()
            self._acquire_compute(ticket, _local.priority)


governor = ResourceGovernor()


def checkpoint():
    current = getattr(_local, 'governor', None)
    if current is None:
        check_cancelled()
    else:
        current.checkpoint()


def estimate_job(path, frame=None):
    # Compressed RAW byte size is only a predecode estimate; actual RSS is
    # checked between stages. Include native model/workspace headroom.
    if frame is not None:
        image_bytes = int(frame.nbytes)
    else:
        try:
            image_bytes = os.path.getsize(path) * 8
        except OSError:
            image_bytes = 128 * MiB
    return max(512 * MiB, image_bytes * 5 + 256 * MiB)


def check_native_memory():
    check_cancelled()
    current = getattr(_local, 'governor', None)
    if current is not None:
        current.check_memory()
