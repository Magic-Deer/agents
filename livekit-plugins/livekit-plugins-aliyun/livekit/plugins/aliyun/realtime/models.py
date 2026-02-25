# Model
BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
DEFAULT_MODEL = "qwen3-omni-flash-realtime"
DEFAULT_VOICE = "Cherry"

# Audio format constants
INPUT_SAMPLE_RATE = 16000  # 16kHz for input
OUTPUT_SAMPLE_RATE = 24000  # 24kHz for output (Flash model)
OUTPUT_SAMPLE_RATE_TURBO = 16000  # 16kHz for output (Turbo model)
NUM_CHANNELS = 1  # Mono
BYTES_PER_SAMPLE = 2  # 16-bit PCM

# Audio chunk size (100ms)
CHUNK_DURATION_MS = 100
CHUNK_SIZE_SAMPLES = INPUT_SAMPLE_RATE * CHUNK_DURATION_MS // 1000
