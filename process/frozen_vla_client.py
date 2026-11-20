"""Use the frozen OpenPI endpoint from the independent policy-learning environment."""

from openpi_client.websocket_client_policy import WebsocketClientPolicy


class FrozenVLAClient:
    def __init__(self, host, port):
        self.client = WebsocketClientPolicy(host, port)
        self.metadata = self.client.get_server_metadata()

    def features(self, observation):
        return self.client.infer({**observation, "dsrl_operation": "features"})["features"]

    def infer(self, observation, *, noise):
        return self.client.infer({**observation, "dsrl_operation": "sample", "dsrl_noise": noise})
