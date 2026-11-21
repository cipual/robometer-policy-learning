"""Native EVA strategy using the unchanged Dense prefix5/H8 execution driver."""
import dataclasses
import time
from core.registry import STRATEGY_REGISTRY
from strategy.base_strategy import BaseInferStrategy


@STRATEGY_REGISTRY.register('DSRLRTCStrategy')
@dataclasses.dataclass
class DSRLRTCStrategy(BaseInferStrategy):
    is_residual_strategy = True
    runs_background_loop: bool = dataclasses.field(default=True, init=False)

    def __post_init__(self):
        super().__post_init__()
        self.client = None
        self.running = False

    def bind_source_rtc(self, client):
        self.client = client

    def prepare_warmup_chunk(self, prompt, fetch_chunk):
        return fetch_chunk(prompt, None)

    def start_loop(self, prompt, fetch_chunk, get_observation):
        self.running = True
        self.last_source_received = time.monotonic()
        self.prepared_time = None

    def control_observation(self, observation, execution_domain):
        try:
            return self._control_observation(observation, execution_domain)
        except Exception as error:
            self.client.execution_error = str(error)
            raise

    def _control_observation(self, observation, execution_domain):
        if execution_domain != 'real':
            raise ValueError('DSRL RTC requires its explicitly configured real execution domain')
        if observation is None:
            if time.monotonic()-self.last_source_received > self.client.metadata['max_source_gap_s']:
                raise ValueError('RTC source observation unavailable beyond the Dense deadline')
            return False
        t = observation['_residual_capture_time']
        if self.client.execution.step == 0 and self.client.execution.active is None:
            if self.prepared_time is None:
                self.client.prepare_initial(observation)
                self.prepared_time = t
                self.last_source_received = time.monotonic()
                return False
            if t <= self.prepared_time:
                if time.monotonic()-self.last_source_received > self.client.metadata['max_source_gap_s']:
                    raise ValueError('RTC initial preparation requires a fresh source before publishing')
                return False
        previous = self.client.execution.last_time
        if previous is not None and t <= previous:
            if time.monotonic()-self.last_source_received > self.client.metadata['max_source_gap_s']:
                raise ValueError('RTC source did not advance')
            return False
        self.last_source_received = time.monotonic()
        self.client.observe(observation)
        return True

    def published_action(self, action):
        self.client.execution.published(action)

    def pop_next_action(self):
        try:
            return self.client.execution.pop()
        except Exception as error:
            self.client.execution_error = str(error)
            raise

    def is_loop_running(self):
        return self.running

    def stop_loop(self):
        self.running = False

    def reset(self):
        super().reset()
        self.running = False
