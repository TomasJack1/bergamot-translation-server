import asyncio
import threading
import wave
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import bergamot
import numpy as np
import sherpa_onnx
import uvicorn
from fastapi import Body, Depends, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

CURRENT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = CURRENT_DIR / "models"

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)


class TranslateManager:
    _service = None
    _lock = threading.Lock()
    _models: ClassVar[dict] = {}

    @classmethod
    async def translate(cls, text: str, src: str = "ja", tgt: str = "zh") -> str:
        """将一种语言翻译成另一种语言

        Args:
            text (str): 要翻译的文本
            src (str, optional): 源语言 Defaults to "ja".
            tgt (str, optional): 目标语言 Defaults to "zh".

        Returns:
            str: 翻译结果
        """
        if f"{src}{tgt}" in cls._models:
            return await cls._translate(text, src, tgt)

        # 如果两种语言不能直接翻译，则采用英语中转
        relay = "en"
        if f"{src}{relay}" in cls._models and f"{relay}{tgt}" in cls._models:
            return await cls._translate(
                await cls._translate(text, src, relay),
                relay,
                tgt,
            )

        return ""

    @classmethod
    async def _translate(cls, text: str, src: str = "ja", tgt: str = "zh") -> str:
        results = await asyncio.create_task(
            asyncio.to_thread(
                cls._service.translate,
                cls._models[f"{src}{tgt}"],
                bergamot.VectorString([text]),
                bergamot.ResponseOptions(),
            ),
        )

        return results[0].target.text

    @classmethod
    def scan_models(cls) -> None:
        """扫描模型目录下所有模型，并加载进内存"""
        for x in MODELS_DIR.iterdir():
            if x.is_dir() and (x / "config.yml").exists():
                cls._models.update(
                    {
                        x.name: cls._service.modelFromConfigPath((x / "config.yml").as_posix()),
                    },
                )

    def __new__(cls):
        if cls._service is None:
            with cls._lock:
                if cls._service is None:
                    cls._service = bergamot.Service(bergamot.ServiceConfig())

        if not cls._models:
            with cls._lock:
                if not cls._models:
                    cls.scan_models()

        return cls


class STTManager:
    _lock = threading.Lock()
    _model = None

    @classmethod
    async def transcribe(cls, audio: bytes) -> str:
        """将语言识别为文字

        Args:
            audio (bytes): 二进制语音数据

        Returns:
            str: 识别结果
        """
        # return await cls._transcribe(audio)
        return await cls._transcribe(audio)

    @classmethod
    async def _transcribe(cls, audio: bytes, language: str = "ja") -> str:
        # 将二进制数据转换为numpy数组
        # audio_np = np.frombuffer(audio, dtype=np.int16)
        # audio_np = audio_np.reshape(-1, 2)
        # audio_np = audio_np[:, 0].astype(np.float32) / 32768.0
        # audio_np = audio_np.mean(axis=1).astype(np.float32) / 32768.0

        # 使用Whisper进行识别
        # 创建内存中的WAV文件
        wav_buffer = BytesIO(audio)

        # 创建WAV文件头
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44800)
            wf.writeframes(audio)

        wav_buffer.seek(0)

        segments, _ = cls._model.transcribe(
            wav_buffer,
            language=language,
            suppress_blank=False,
            vad_filter=True,
        )

        return await asyncio.create_task(
            asyncio.to_thread(
                lambda segs: "".join(seg.text for seg in segs),
                segments,
            ),
        )

    @staticmethod
    def get_text(segs):
        return "".join(seg.text for seg in segs)

    def __new__(cls):
        if not cls._model:
            with cls._lock:
                if not cls._model:
                    cls._model = WhisperModel("./models/faster-whisper-tiny", device="cpu")
                    # , compute_type="int8"

        return cls


class SherpaManager:
    _lock = threading.Lock()
    _model = None

    @classmethod
    async def transcribe(cls, audio: bytes) -> str:
        """将语言识别为文字

        Args:
            audio (bytes): 二进制语音数据

        Returns:
            str: 识别结果
        """
        # return await cls._transcribe(audio)
        return await cls._transcribe(audio)

    @classmethod
    async def _transcribe(cls, audio: bytes, language: str = "ja") -> str:
        # 使用Whisper进行识别
        # 创建内存中的WAV文件
        # wav_buffer = BytesIO(audio)

        # 创建WAV文件头
        # with wave.open(wav_buffer, "wb") as wf:
        #     wf.setnchannels(2)
        #     wf.setsampwidth(2)
        #     wf.setframerate(48000)
        #     wf.writeframes(audio)

        # wav_buffer.seek(0)

        # 将二进制数据转换为numpy数组
        audio_np = np.frombuffer(audio, dtype=np.int16)
        audio_np = audio_np.astype(np.float32) / 32768.0

        streams = []
        s = cls._model.create_stream()
        s.accept_waveform(48000, audio_np)
        streams.append(s)

        def decode_streams(streams):
            cls._model.decode_streams(streams)
            return "".join([s.result.text for s in streams])

        return await asyncio.create_task(
            asyncio.to_thread(decode_streams, streams),
        )

    def __new__(cls):
        if not cls._model:
            with cls._lock:
                if not cls._model:
                    cls._model = sherpa_onnx.OfflineRecognizer.from_transducer(
                        tokens="./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/tokens.txt",
                        encoder="./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/encoder-epoch-99-avg-1.onnx",
                        decoder="./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/decoder-epoch-99-avg-1.onnx",
                        joiner="./models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01/joiner-epoch-99-avg-1.onnx",
                        num_threads=4,
                        provider="cpu",
                        sample_rate=48000,
                        feature_dim=80,
                        decoding_method="greedy_search",
                        max_active_paths=4,
                        lm="",
                        lm_scale=0.1,
                        hotwords_file="",
                        hotwords_score=15,
                        modeling_unit="",
                        bpe_vocab="",
                        blank_penalty=0.1,
                    )

        return cls


@app.post("/translate")
async def translate(
    text: str = Body(embed=True),
    translator: TranslateManager = Depends(TranslateManager),
):
    return {"result": await translator.translate(text)}


@app.websocket("/ws")
async def ws(
    websocket: WebSocket,
    # asr: STTManager = Depends(STTManager),
    asr: SherpaManager = Depends(SherpaManager),
):
    await websocket.accept()

    last_data = b""
    s = ""

    # async for data in websocket.iter_bytes():
    #     result = await asr.transcribe(last_data + data)

    #     if not result:
    #         last_data += data
    #     else:
    #         last_data = b""
    #         s += result

    #         print(s)
    async for data in websocket.iter_bytes():
        last_data += data

    result = await asr.transcribe(last_data)

    print(result)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
