"""One cancellable denoise job; the governor lends its tiles to previews."""
import threading

from raw_alchemy.pipeline.cancellation import cancellation_scope
from raw_alchemy.pipeline.executor import PipelineAborted
from raw_alchemy.pipeline.resources import governor, estimate_job
from raw_alchemy.pipeline.source_artifacts import resolve_denoised_source


class DenoiseTask(threading.Thread):
    def __init__(self, path, strength, source, source_token, policy_token,
                 decode_variant, *, denoise, started, progress, finished, wake):
        super().__init__(name='RawAlchemy-Denoise', daemon=False)
        self.path = path
        self.strength = strength
        self.source = source
        self.source_token = source_token
        self.policy_token = policy_token
        self.decode_variant = decode_variant
        self.denoise = denoise
        self.started = started
        self.progress = progress
        self.finished = finished
        self.wake = wake
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.ready = threading.Event()
        self.published = False  # written only by the preview lane
        self.result = None
        self.error = None

    @property
    def key(self):
        return (self.path, self.strength, self.source_token, self.policy_token)

    def run(self):
        announced = False
        notified = False

        def ready(result):
            nonlocal notified
            self.result = result
            self.ready.set()
            self.finished()
            notified = True
            self.wake.set()

        try:
            with cancellation_scope(self.cancelled.is_set):
                with governor.job(estimate_job(self.path, self.source), priority=2):
                    self.started()
                    announced = True
                    self.result = resolve_denoised_source(
                        self.path, self.strength, source=self.source,
                        decode_variant=self.decode_variant, denoise=self.denoise,
                        should_abort=self.cancelled.is_set,
                        progress_callback=self.progress,
                        expected_source_token=self.source_token,
                        result_callback=ready,
                    )
        except PipelineAborted:
            pass
        except Exception as exc:
            self.error = exc
        finally:
            if announced and not notified:
                self.finished()
            self.done.set()
            self.wake.set()
