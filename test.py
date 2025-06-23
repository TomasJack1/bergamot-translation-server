# import time

# import numpy as np
# from faster_whisper import WhisperModel

# model = WhisperModel("./models/faster-whisper-tiny", device="cpu", compute_type="int8")

# # 将二进制数据转换为numpy数组
# # audio_np = np.frombuffer(result, dtype=np.int16).astype(np.float32) / 32768.0

# # 使用Whisper进行识别
# segments, info = model.transcribe(
#     "output.wav",
#     language="ja",
#     suppress_blank=False,
#     without_timestamps=True,
#     vad_filter=True,
# )


# start_time = time.time()

# print(list(segments))
# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
# print(time.time() - start_time)

import av

with av.open(__import__("io").BytesIO(), mode="r") as file:
    pass
