import json, asyncio, logging
from rgws.interface import WebsocketServer
from sound_generator import get_prediction

logging.basicConfig(level=logging.DEBUG)


class SGServerInterface(WebsocketServer):
    def __init__(self, **kwargs):
        super(SGServerInterface, self).__init__(**kwargs)
        self._register(self.get_prediction)

    async def _consumer(self, websocket, message):
        ret = await self.dispatch(message)
        async for gen in ret:
            await websocket.send(gen)

    async def get_prediction(self, data):
        ret = get_prediction(data).tolist()
        data = json.dumps({"resp": ret})
        return self.make_data_stream(data)


if __name__ == "__main__":
    s = SGServerInterface(host="localhost", port=8080)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(s.run())
    loop.run_forever()
