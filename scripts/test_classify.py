"""Quick smoke test for /classify — run inside container."""
import io
import urllib.request
from PIL import Image

buf = io.BytesIO()
Image.new("RGB", (100, 100), (0, 255, 0)).save(buf, "JPEG")
img = buf.getvalue()

boundary = "BOUNDARY"
body = (
    f'--{boundary}\r\nContent-Disposition: form-data; name="model_choice"\r\n\r\nGiga 2\r\n'.encode()
    + f'--{boundary}\r\nContent-Disposition: form-data; name="image_file"; filename="test.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
    + img
    + f"\r\n--{boundary}--\r\n".encode()
)

req = urllib.request.Request("http://127.0.0.1:8000/classify", data=body, method="POST")
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")

with urllib.request.urlopen(req) as resp:
    print("status", resp.status)
