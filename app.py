import os
import sys
import torch
import shutil
import librosa
import warnings
import subprocess
import numpy as np
import gradio as gr
import librosa.display
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from collections import Counter
from model import EvalNet
from utils import (
    get_modelist,
    find_mp3_files,
    _L,
    MODEL_DIR,
    CACHE_DIR,
    TRANSLATE,
    CLASSES,
)


def most_common_element(input_list):
    counter = Counter(input_list)
    mce, _ = counter.most_common(1)[0]
    return mce


def mp3_to_mel(audio_path: str, width=11.4):
    y, sr = librosa.load(audio_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    dur = librosa.get_duration(y=y, sr=sr)
    total_frames = log_mel_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_mel_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{CACHE_DIR}/mel_{round(dur, 2)}_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def mp3_to_cqt(audio_path: str, width=11.4):
    y, sr = librosa.load(audio_path)
    cqt_spec = librosa.cqt(y=y, sr=sr)
    log_cqt_spec = librosa.power_to_db(np.abs(cqt_spec) ** 2, ref=np.max)
    dur = librosa.get_duration(y=y, sr=sr)
    total_frames = log_cqt_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_cqt_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{CACHE_DIR}/cqt_{round(dur, 2)}_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def mp3_to_chroma(audio_path: str, width=11.4):
    y, sr = librosa.load(audio_path)
    chroma_spec = librosa.feature.chroma_stft(y=y, sr=sr)
    log_chroma_spec = librosa.power_to_db(np.abs(chroma_spec) ** 2, ref=np.max)
    dur = librosa.get_duration(y=y, sr=sr)
    total_frames = log_chroma_spec.shape[1]
    step = int(width * total_frames / dur)
    count = int(total_frames / step)
    begin = int(0.5 * (total_frames - count * step))
    end = begin + step * count
    for i in range(begin, end, step):
        librosa.display.specshow(log_chroma_spec[:, i : i + step])
        plt.axis("off")
        plt.savefig(
            f"{CACHE_DIR}/chroma_{round(dur, 2)}_{i}.jpg",
            bbox_inches="tight",
            pad_inches=0.0,
        )
        plt.close()


def embed_img(img_path, input_size=224):
    transform = transforms.Compose(
        [
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


def infer(mp3_path, log_name: str, folder_path=CACHE_DIR):
    status = "Success"
    filename = result = None
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        if not mp3_path:
            raise ValueError("请输入音频!")

        spec = log_name.split("_")[-1]
        os.makedirs(folder_path, exist_ok=True)
        network = EvalNet(log_name)
        eval("mp3_to_%s" % spec)(mp3_path)
        outputs = []
        all_files = os.listdir(folder_path)
        for file_name in all_files:
            if file_name.lower().endswith(".jpg"):
                file_path = os.path.join(folder_path, file_name)
                input = embed_img(file_path)
                output: torch.Tensor = network.model(input)
                pred_id = torch.max(output.data, 1)[1]
                outputs.append(int(pred_id))

        max_count_item = most_common_element(outputs)
        filename = os.path.basename(mp3_path)
        result = TRANSLATE[CLASSES[max_count_item]]

    except Exception as e:
        status = f"{e}"

    return status, filename, result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ffmpeg = "ffmpeg-release-amd64-static"
    if sys.platform.startswith("linux"):
        if not os.path.exists(f"./{ffmpeg}.tar.xz"):
            shutil.move(os.path.abspath(f"{MODEL_DIR}/{ffmpeg}"), f"./{ffmpeg}.tar.xz")

        folder_path = f"{os.getcwd()}/{ffmpeg}"
        if not os.path.exists(folder_path):
            subprocess.call(f"tar -xvf {ffmpeg}.tar.xz", shell=True)

        os.environ["PATH"] = f"{folder_path}:{os.environ.get('PATH', '')}"

    models = get_modelist(assign_model="vgg19_bn_cqt")
    examples = []
    example_mp3s = find_mp3_files()
    for mp3 in example_mp3s:
        examples.append([mp3, models[0]])

    with gr.Blocks() as demo:
        gr.Interface(
            fn=infer,
            inputs=[
                gr.Audio(label=_L("上传 MP3 音频"), type="filepath"),
                gr.Dropdown(choices=models, label=_L("选择模型"), value=models[0]),
            ],
            outputs=[
                gr.Textbox(label=_L("状态栏"), buttons=["copy"]),
                gr.Textbox(label=_L("音频文件名"), buttons=["copy"]),
                gr.Textbox(label=_L("流派识别"), buttons=["copy"]),
            ],
            examples=examples,
            cache_examples=False,
            flagging_mode="never",
            title=_L("建议录音时长保持在 15s 以内, 过长会影响识别效率"),
        )

        gr.Markdown(f"# {_L('引用')}" + """
            ```bibtex
            @dataset{zhaorui_liu_2021_5676893,
                author    = {Zhaorui Liu and Zijin Li},
                title     = {Music Data Sharing Platform for Computational Musicology Research (CCMUSIC DATASET)},
                month     = nov,
                year      = 2021,
                publisher = {Zenodo},
                version   = {1.1},
                doi       = {10.5281/zenodo.5676893},
                url       = {https://doi.org/10.5281/zenodo.5676893}
            }
            ```""")

    demo.launch(css="#gradio-share-link-button-0 { display: none; }", ssr_mode=False)
