cat > server/arduino_endpoints.py <<'PY'
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any
import time
import asyncio

router = APIRouter()

# In-memory state (small and simple)
SENSOR_HISTORY: List[Dict[str,Any]] = []
LAST_STATUS = {"online": False, "updated_at": None}
LAST_CALIBRATION = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active.append(websocket)

    def disconnect(self, websocket: WebSocket):
        try:
            self.active.remove(websocket)
        except ValueError:
            pass

    async def broadcast(self, message: dict):
        living = []
        for ws in self.active:
            try:
                await ws.send_json(message)
                living.append(ws)
            except Exception:
                pass
        self.active = living

manager = ConnectionManager()

class SensorPayload(BaseModel):
    ts: float
    raw: Dict[str, float]
    corrected: Dict[str, float]
    calibrated: bool = True

class StatusPayload(BaseModel):
    online: bool

class CalibrationPayload(BaseModel):
    calibrated_at: float
    avg_raw: Dict[str, float]
    correction: Dict[str, float]
    lab_values: Dict[str, float]

@router.post("/api/sensor-data")
async def receive_sensor(payload: SensorPayload):
    entry = payload.dict()
    entry["received_at"] = time.time()
    SENSOR_HISTORY.append(entry)
    # broadcast
    await manager.broadcast({"type": "sensor", "data": entry})
    return {"ok": True}

@router.post("/api/status")
async def set_status(payload: StatusPayload):
    LAST_STATUS["online"] = payload.online
    LAST_STATUS["updated_at"] = time.time()
    await manager.broadcast({"type": "status", "data": LAST_STATUS})
    return {"ok": True}

@router.post("/api/calibration")
async def set_calibration(payload: CalibrationPayload):
    global LAST_CALIBRATION
    LAST_CALIBRATION = payload.dict()
    await manager.broadcast({"type": "calibration", "data": LAST_CALIBRATION})
    return {"ok": True}

@router.get("/api/history")
async def get_history(limit: int = 100):
    return SENSOR_HISTORY[-limit:]

# WebSocket endpoint path for frontend clients
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # send initial data
        await websocket.send_json({"type": "status", "data": LAST_STATUS})
        if LAST_CALIBRATION:
            await websocket.send_json({"type": "calibration", "data": LAST_CALIBRATION})
        await websocket.send_json({"type": "history", "data": SENSOR_HISTORY[-50:]})
        while True:
            # read to detect disconnects (clients don't need to send)
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)
PY