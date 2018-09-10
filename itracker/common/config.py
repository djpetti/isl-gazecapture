import network

# Model to use for prediction.
MODEL_FILE = "models/eye_model_large.hd5"
# Port to listen on for the server.
SERVER_PORT = 6219

# Minimum confidence before we ignore detections.
MIN_CONFIDENCE = 0.20

# Specifies the network architecture we will use.
NET_ARCH = network.ResidualNetwork
# The shape of raw images from the dataset.
RAW_SHAPE = (400, 400, 3)
# The shape of the input face images to the network.
FACE_SHAPE = (224, 224, 3)
# The shape of the input eye images to the network.
EYE_SHAPE = (224, 224, 1)
# Time in seconds that an image gets to live in the pipeline before we consider
# it stale.
STALE_THRESHOLD = 0.1
