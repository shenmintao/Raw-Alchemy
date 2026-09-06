"""One bounded export lane; the shared governor serializes expensive stages."""
import threading

from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.resources import governor, estimate_job


class ExportDispatcher(threading.Thread):
    def __init__(self, owner):
        super().__init__(name='RawAlchemy-Export', daemon=False)
        self.owner = owner
        self.wake = threading.Event()

    def run(self):
        owner = self.owner
        while not owner._should_stop:
            with owner.lock:
                request = owner._export_queue.pop(0) if owner._export_queue else None
                if request is None:
                    self.wake.clear()
            if request is None:
                self.wake.wait(0.1)
                continue
            try:
                frame = request.params['export'].get('cached_img')
                with cancellation_scope(lambda: owner._should_stop):
                    with governor.job(estimate_job(request.path, frame), priority=1):
                        owner._do_export(request)
            except Exception as exc:
                owner.export_completed.emit(request.request_id, False, str(exc))
                owner.export_finished.emit(False, str(exc))
