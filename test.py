from pathlib import Path


for p in Path(__file__).parent.resolve().iterdir():
    if p.is_dir():
        print(str(p))
