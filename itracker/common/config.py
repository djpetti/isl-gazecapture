import network

# Model to use for prediction.
MODEL_FILE = "models/eye_model_large.hd5"
# Port to listen on for the server.
SERVER_PORT = 6219

# Minimum confidence before we ignore detections.
MIN_CONFIDENCE = 0.20

# Specifies the network architecture we will use.
NET_ARCH = network.LargeNetwork
# Time in seconds that an image gets to live in the pipeline before we consider
# it stale.
STALE_THRESHOLD = 0.1
