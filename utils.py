import os
import torch

EN_US = os.getenv("LANG") != "zh_CN.UTF-8"

ZH2EN = {
    "上传 MP3 音频": "Upload MP3",
    "选择模型": "Select a model",
    "状态栏": "Status",
    "音频文件名": "Audio filename",
    "流派识别": "Genre recognition",
    "建议录音时长保持在 15s 以内, 过长会影响识别效率": "It is recommended to keep the duration of recording within 15s, too long will affect the recognition efficiency.",
    "引用": "Cite",
    "交响乐": "Symphony",
    "戏曲": "Opera",
    "独奏": "Solo",
    "室内乐": "Chamber",
    "芭乐": "Pop vocal ballad",
    "成人时代": "Adult contemporary",
    "青少年流行": "Teen pop",
    "当代流行舞曲": "Contemporary dance pop",
    "流行舞曲": "Dance pop",
    "经典独立流行": "Classic indie pop",
    "室内卡巴莱与艺术流行乐": "Chamber cabaret & art pop",
    "灵魂乐或节奏布鲁斯": "Soul / R&B",
    "成人另类摇滚": "Adult alternative rock",
    "迷幻民族摇滚": "Uplifting anthemic rock",
    "慢摇滚": "Soft rock",
    "原声流行": "Acoustic pop",
}

if EN_US:
    import huggingface_hub

    MODEL_DIR = huggingface_hub.snapshot_download(
        "ccmusic-database/music_genre",
        cache_dir="./__pycache__",
    )

else:
    import modelscope

    MODEL_DIR = modelscope.snapshot_download(
        "ccmusic-database/music_genre",
        cache_dir="./__pycache__",
    )


def _L(zh_txt: str):
    return ZH2EN[zh_txt] if EN_US else zh_txt


TRANSLATE = {
    "Symphony": _L("交响乐"),
    "Opera": _L("戏曲"),
    "Solo": _L("独奏"),
    "Chamber": _L("室内乐"),
    "Pop_vocal_ballad": _L("芭乐"),
    "Adult_contemporary": _L("成人时代"),
    "Teen_pop": _L("青少年流行"),
    "Contemporary_dance_pop": _L("当代流行舞曲"),
    "Dance_pop": _L("流行舞曲"),
    "Classic_indie_pop": _L("经典独立流行"),
    "Chamber_cabaret_and_art_pop": _L("室内卡巴莱与艺术流行乐"),
    "Soul_or_r_and_b": _L("灵魂乐或节奏布鲁斯"),
    "Adult_alternative_rock": _L("成人另类摇滚"),
    "Uplifting_anthemic_rock": _L("迷幻民族摇滚"),
    "Soft_rock": _L("慢摇滚"),
    "Acoustic_pop": _L("原声流行"),
}
CLASSES = list(TRANSLATE.keys())
CACHE_DIR = "./__pycache__/tmp"


def toCUDA(x):
    if hasattr(x, "cuda"):
        if torch.cuda.is_available():
            return x.cuda()

    return x


def find_mp3_files(folder_path=f"{MODEL_DIR}/examples"):
    wav_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".mp3"):
                file_path = os.path.join(root, file)
                wav_files.append(file_path)

    return wav_files


def get_modelist(model_dir=MODEL_DIR, assign_model=""):
    output = []
    for entry in os.listdir(model_dir):
        # 获取完整路径
        full_path = os.path.join(model_dir, entry)

        # 跳过'.git'文件夹
        if entry == ".git" or entry == "examples":
            print(f"跳过 .git / examples 文件夹: {full_path}")
            continue

        # 检查条目是文件还是目录
        if os.path.isdir(full_path):
            model = os.path.basename(full_path)
            if assign_model and assign_model.lower() in model:
                output.insert(0, model)
            else:
                output.append(model)

    return output
