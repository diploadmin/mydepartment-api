from fastapi import WebSocket
import json

# this controls websocket connections and messages that are returned to the frontend
# connect
# disconnect
# close all connections
# send message
class WebSocketManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        try:
            await websocket.close()
        except Exception:
            pass

    async def close_all_connections(self):
        for websocket in self.active_connections:
            await websocket.close()
        self.active_connections.clear()

    async def send_message(self, message: str, websocket: WebSocket):
        
        try:
            # Attempt to serialize the message to JSON
            message_to_send = json.dumps(message)
        except TypeError as e:
            # If the message cannot be serialized to JSON, raise an error
            raise ValueError(f"Message cannot be serialized to JSON: {e}")

        await websocket.send_text(message_to_send)

    # async def broadcast(self, message: str):
    #     for connection in self.active_connections:
    #         await connection.send_text(message)
