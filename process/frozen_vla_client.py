"""Use the frozen OpenPI endpoint from the independent policy-learning environment."""

from openpi_client.websocket_client_policy import WebsocketClientPolicy


class FrozenVLAClient:
    def __init__(self, host, port):
        self.client = WebsocketClientPolicy(host, port)
        self.metadata = self.client.get_server_metadata()

    def features(self, observation, *, prefix=None):
        request = {**observation, 'dsrl_operation': 'features'}
        if prefix is not None:
            request['rtc_prefix'] = prefix
        return self.client.infer(request)['features']

    def prepare(self, observation, *, prefix=None):
        """Encode one source observation and retain its one-use sampling context."""
        request = {**observation, 'dsrl_operation': 'prepare'}
        if prefix is not None:
            request['rtc_prefix'] = prefix
        response = self.client.infer(request)
        return response["features"], response["context_id"]

    def infer(self, observation, *, noise, baseline_noise=None):
        request = {"dsrl_context_id": observation["dsrl_context_id"],
                   "dsrl_operation": "sample_prepared"}
        if baseline_noise is not None:
            request["dsrl_baseline_noise"] = baseline_noise
        return self.client.infer({**request, "dsrl_noise": noise})
