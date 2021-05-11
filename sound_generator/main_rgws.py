import asyncio, logging
from rgws.interface import WebsocketServer
from sound_generator import get_prediction

logging.basicConfig(level=logging.DEBUG)


class SGServerInterface(WebsocketServer):
    def __init__(self, **kwargs):
        super(SGServerInterface, self).__init__(**kwargs)
        self.state = "Initializing"
        self._register(self.get_prediction)
        self._register(self.setup_model)
        self._register(self.status)
        self.state = "Initialized"

    async def _consumer(self, ws, message):
        ret = await self.dispatch(message)
        async for gen in ret:
            await ws.send_json(gen)

    async def get_prediction(self, data):
        self.state = "Processing"
        resp, success = get_prediction(data)
        yield {"resp": resp.tolist(), "success": success}
        self.state = "Processed"

    async def setup_model(self):
        yield {"resp": True}

    async def status(self):
        yield {"resp": self.state}


if __name__ == "__main__":
    s = SGServerInterface(host="localhost", port=8080)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(s.run())
    loop.run_forever()
