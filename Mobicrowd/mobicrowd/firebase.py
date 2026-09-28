import os, json, base64
import firebase_admin
from firebase_admin import credentials

def ensure_firebase():
    """Init Firebase Admin exactly once from a base64 env var."""
    if firebase_admin._apps:
        return
    b64 = os.getenv("FIREBASE_SERVICE_ACCOUNT_B64")
    if not b64:
        raise RuntimeError("Set FIREBASE_SERVICE_ACCOUNT_B64")
    data = json.loads(base64.b64decode(b64).decode("utf-8"))
    cred = credentials.Certificate(data)
    firebase_admin.initialize_app(cred)
