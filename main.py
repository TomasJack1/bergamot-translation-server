import asyncio
import threading
from pathlib import Path
from typing import ClassVar

import bergamot
import uvicorn
from fastapi import Body, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.post("/translate")
async def translate(
    text: str = Body(embed=True),
    translator: TranslateManager = Depends(TranslateManager),
):
    return {"result": await translator.translate(text)}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
