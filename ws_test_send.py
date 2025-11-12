import asyncio, json, random, time
import websockets

async def main():
    uri = "ws://127.0.0.1:8000/ws/esp32"
    async with websockets.connect(uri) as ws:
        for i in range(5):
            payload = {
                "N": random.randint(20,120),
                "P": random.randint(10,100),
                "K": random.randint(20,150),
                "pH": round(6 + random.random(),2),
                "temp": round(24 + random.random()*5,2),
                "humidity": round(40 + random.random()*40,1),
                "soil": round(30 + random.random()*40,1),
                "rainfall": round(random.random()*5,2),
                "lat": 12.9716,
                "lon": 77.5946,
                "ts": int(time.time()*1000)
            }
            await ws.send(json.dumps(payload))
            print("Sent:", payload)
            try:
                echo = await asyncio.wait_for(ws.recv(), timeout=0.5)
                print("Echo:", echo)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.5)

asyncio.run(main())