import time

import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "outputs/checkpoints/unet_raw.onnx",
    providers=["CPUExecutionProvider"],
)

dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)

# warmup
for _ in range(10):
    session.run(None, {"input": dummy_input})

times = []

for _ in range(100):

    start = time.time()

    outputs = session.run(
        None,
        {"input": dummy_input},
    )

    end = time.time()

    times.append(end - start)

avg_ms = np.mean(times) * 1000

fps = 1000 / avg_ms

print(f"Average latency: {avg_ms:.2f} ms")
print(f"FPS: {fps:.2f}")