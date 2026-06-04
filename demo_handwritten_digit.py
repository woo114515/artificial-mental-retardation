"""
Browser demo for drawing a digit and predicting it with a saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from config import CHECKPOINT_PATH, IMAGE_HEIGHT, IMAGE_WIDTH
from train import build_model, prepare_data
from utils.checkpoint import load_checkpoint
from utils.prediction_visualization import softmax


HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Handwritten Digit Demo</title>
  <style>
    :root {
      color-scheme: light;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f8;
      color: #171a1f;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    main {
      width: min(92vw, 760px);
      display: grid;
      grid-template-columns: minmax(280px, 340px) 1fr;
      gap: 24px;
      align-items: start;
    }
    h1 {
      grid-column: 1 / -1;
      margin: 0;
      font-size: 28px;
      font-weight: 700;
      letter-spacing: 0;
    }
    canvas {
      width: 320px;
      height: 320px;
      max-width: 100%;
      aspect-ratio: 1;
      touch-action: none;
      cursor: crosshair;
      background: #000;
      border: 1px solid #d4d7dd;
      border-radius: 8px;
      box-shadow: 0 10px 30px rgba(23, 26, 31, 0.10);
    }
    .controls {
      display: flex;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      border: 1px solid #c9cdd4;
      background: #fff;
      color: #171a1f;
      border-radius: 6px;
      padding: 10px 14px;
      font-size: 15px;
      cursor: pointer;
    }
    button.primary {
      background: #171a1f;
      border-color: #171a1f;
      color: white;
    }
    button:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .panel {
      display: grid;
      gap: 16px;
    }
    .result {
      font-size: 80px;
      line-height: 1;
      font-weight: 750;
      min-height: 90px;
    }
    .confidence {
      font-size: 18px;
      min-height: 26px;
    }
    .bars {
      display: grid;
      gap: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: 22px 1fr 52px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }
    .bar {
      height: 10px;
      border-radius: 999px;
      background: #e3e6ea;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      width: 0%;
      background: #171a1f;
      transition: width 140ms ease-out;
    }
    .meta {
      font-size: 12px;
      color: #626975;
      line-height: 1.5;
      word-break: break-word;
    }
    @media (max-width: 720px) {
      main {
        grid-template-columns: 1fr;
      }
      h1 {
        font-size: 24px;
      }
    }
  </style>
</head>
<body>
  <main>
    <h1>Handwritten Digit Demo</h1>
    <section>
      <canvas id="canvas" width="280" height="280" aria-label="drawing canvas"></canvas>
      <div class="controls">
        <button class="primary" id="predict">预测</button>
        <button id="clear">清空</button>
      </div>
    </section>
    <section class="panel">
      <div class="result" id="digit">-</div>
      <div class="confidence" id="confidence"></div>
      <div class="bars" id="bars"></div>
      <div class="meta" id="meta"></div>
    </section>
  </main>
  <script>
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const predictButton = document.getElementById("predict");
    const clearButton = document.getElementById("clear");
    const digit = document.getElementById("digit");
    const confidence = document.getElementById("confidence");
    const bars = document.getElementById("bars");
    const meta = document.getElementById("meta");
    let drawing = false;
    let lastPoint = null;

    function resetCanvas() {
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      digit.textContent = "-";
      confidence.textContent = "";
      bars.innerHTML = "";
    }

    function pointFromEvent(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        x: (event.clientX - rect.left) * canvas.width / rect.width,
        y: (event.clientY - rect.top) * canvas.height / rect.height,
      };
    }

    function drawLine(from, to) {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 22;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }

    canvas.addEventListener("pointerdown", (event) => {
      drawing = true;
      lastPoint = pointFromEvent(event);
      drawLine(lastPoint, lastPoint);
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!drawing) return;
      const nextPoint = pointFromEvent(event);
      drawLine(lastPoint, nextPoint);
      lastPoint = nextPoint;
    });

    canvas.addEventListener("pointerup", () => {
      drawing = false;
      lastPoint = null;
    });

    canvas.addEventListener("pointercancel", () => {
      drawing = false;
      lastPoint = null;
    });

    function getMnistPixels() {
      const small = document.createElement("canvas");
      small.width = 28;
      small.height = 28;
      const smallCtx = small.getContext("2d", { willReadFrequently: true });
      smallCtx.fillStyle = "black";
      smallCtx.fillRect(0, 0, 28, 28);
      smallCtx.imageSmoothingEnabled = true;
      smallCtx.imageSmoothingQuality = "high";
      smallCtx.drawImage(canvas, 0, 0, 28, 28);

      const image = smallCtx.getImageData(0, 0, 28, 28).data;
      const pixels = [];
      for (let i = 0; i < image.length; i += 4) {
        pixels.push(image[i] / 255);
      }
      return pixels;
    }

    function renderProbabilities(probabilities) {
      bars.innerHTML = "";
      probabilities.forEach((prob, index) => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
          <span>${index}</span>
          <span class="bar"><span class="fill" style="width: ${(prob * 100).toFixed(1)}%"></span></span>
          <span>${(prob * 100).toFixed(1)}%</span>
        `;
        bars.appendChild(row);
      });
    }

    async function predictDigit() {
      predictButton.disabled = true;
      confidence.textContent = "Predicting...";
      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pixels: getMnistPixels() }),
        });
        if (!response.ok) {
          throw new Error(await response.text());
        }
        const result = await response.json();
        digit.textContent = result.prediction;
        confidence.textContent = `confidence ${(result.confidence * 100).toFixed(1)}%`;
        meta.textContent = result.metadata_title;
        renderProbabilities(result.probabilities);
      } catch (error) {
        confidence.textContent = error.message;
      } finally {
        predictButton.disabled = false;
      }
    }

    clearButton.addEventListener("click", resetCanvas);
    predictButton.addEventListener("click", predictDigit);
    resetCanvas();
  </script>
</body>
</html>
"""


def prepare_pixels(pixels: list[float]) -> np.ndarray:
    """
    Convert 28x28 canvas grayscale pixels to one model input sample.
    """
    X = np.asarray(pixels, dtype=np.float32)

    if X.shape != (IMAGE_HEIGHT * IMAGE_WIDTH,):
        raise ValueError(
            f"Expected {IMAGE_HEIGHT * IMAGE_WIDTH} pixels, got shape {X.shape}."
        )

    if np.max(X) > 1.0:
        X = X / 255.0

    X = np.clip(X, 0.0, 1.0).reshape(1, -1)
    X_model, _, _ = prepare_data(X, X, X)
    return X_model


def metadata_title(metadata: dict) -> str:
    keys = [
        "MODEL_TYPE",
        "OPTIMIZER",
        "ACTIVATION",
        "LEARNING_RATE",
        "BATCH_SIZE",
        "HIDDEN_DIMS",
        "WEIGHT_INIT",
        "RANDOM_SEED",
    ]
    if metadata.get("MODEL_TYPE") == "cnn":
        keys.extend(["CNN_OUT_CHANNELS", "CNN_KERNEL_SIZE"])

    parts = [f"{key}={metadata[key]}" for key in keys if key in metadata]
    return ", ".join(parts) if parts else "checkpoint metadata unavailable"


def create_handler(model, metadata):
    class DemoHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path

            if path in {"/", "/index.html"}:
                self._send_html(HTML)
                return

            if path == "/health":
                self._send_json({"ok": True})
                return

            self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path

            if path != "/predict":
                self.send_error(404)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                X_model = prepare_pixels(payload["pixels"])

                logits = model.forward(X_model)
                probabilities = softmax(logits)[0]
                prediction = int(np.argmax(probabilities))
                confidence = float(probabilities[prediction])

                self._send_json(
                    {
                        "prediction": prediction,
                        "confidence": confidence,
                        "probabilities": probabilities.tolist(),
                        "metadata": metadata,
                        "metadata_title": metadata_title(metadata),
                    }
                )
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self.send_error(400, str(error))

        def log_message(self, format, *args):
            return

        def _send_html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return DemoHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a browser handwritten digit demo.")
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_PATH,
        help=f"Checkpoint path to load. Default: {CHECKPOINT_PATH}",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for the demo server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the demo server.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model()
    metadata = load_checkpoint(model, args.checkpoint)

    handler = create_handler(model, metadata)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Loaded checkpoint: {Path(args.checkpoint)}")
    print(f"Checkpoint metadata: {metadata}")
    print(f"Open http://{args.host}:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
