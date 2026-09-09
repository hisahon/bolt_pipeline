"""
frames_to_timelapse.py

Agrega os PNGs ja salvos em cada subpasta de IMAGE_DIR (uma por dia, geradas
por lightning_goes_trajec.py) e monta um video .mp4 por dia, na ordem
cronologica -- o proprio nome do arquivo ja ordena certo
(ex.: 20260723_DOY204_1230_CH13.png).

Nao gera nenhuma figura nova -- so junta em video as que ja existem.
Arquivos soltos dentro de IMAGE_DIR (que nao sejam subpastas) sao ignorados.

Requisitos:
    pip install imageio imageio-ffmpeg
"""

from pathlib import Path
import imageio.v2 as imageio

IMAGE_DIR = Path(r"D:\bolt_pipeline\data\images")
VIDEO_DIR = Path(r"D:\bolt_pipeline\data\videos")
FPS       = 2


def main():
    if not IMAGE_DIR.exists():
        print(f"[ERRO] {IMAGE_DIR} nao existe.")
        return

    day_dirs = sorted(p for p in IMAGE_DIR.iterdir() if p.is_dir())
    if not day_dirs:
        print(f"Nenhuma subpasta encontrada em {IMAGE_DIR}.")
        return

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    for day_dir in day_dirs:
        frames = sorted(day_dir.glob("*.png"))
        if not frames:
            print(f"[{day_dir.name}] nenhum PNG, pulando.")
            continue

        out_path = VIDEO_DIR / f"{day_dir.name}.mp4"
        print(f"[{day_dir.name}] {len(frames)} frame(s) -> {out_path.name}")

        writer = imageio.get_writer(
            out_path, format="FFMPEG", fps=FPS, macro_block_size=None
        )
        try:
            for f in frames:
                img = imageio.imread(f)
                h, w = img.shape[:2]
                # libx264 exige altura/largura pares -- corta 1px de sobra se for impar
                img = img[: h - (h % 2), : w - (w % 2)]
                writer.append_data(img)
        finally:
            writer.close()

    print("\nTudo pronto.")


if __name__ == "__main__":
    main()