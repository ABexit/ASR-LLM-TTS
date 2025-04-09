from transformers import AutoModelForCausalLM, AutoTokenizer
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
from funasr import AutoModel
import torchaudio
import pygame
import time
import sys
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(filename="output.wav", sample_rate=44100):
    print("按下 Enter 开始录音...")
    input()  # 等待用户按下 Enter 键开始录音
    print("录音中... 按下 Enter 键结束录音")
    
    # 开始录音
    recording = []
    try:
        def callback(indata, frames, time, status):
            recording.append(indata.copy())
        with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback):
            input()  # 等待用户再次按下 Enter 键结束录音
    except Exception as e:
        print(f"录音出现错误: {e}")
        return
    
    # 将录音数据合并并保存为 WAV 文件
    audio_data = np.concatenate(recording, axis=0)
    write(filename, sample_rate, (audio_data * 32767).astype(np.int16))
    print(f"录音已保存为 {filename}")


# --- 播放音频 -
def play_audio(file_path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)  # 等待音频播放结束
        print("播放完成！")
    except Exception as e:
        print(f"播放失败: {e}")
    finally:
        pygame.mixer.quit()

import os
import shutil

def clear_folder(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"文件夹 '{folder_path}' 不存在，已创建")
        return
    
    # 获取文件夹中的所有文件和子文件夹
    items = os.listdir(folder_path)
    
    # 如果文件夹为空，直接返回
    if not items:
        print(f"文件夹 '{folder_path}' 已经为空")
        return
    
    # 遍历文件和文件夹并删除
    for item in items:
        item_path = os.path.join(folder_path, item)
        
        # 判断是否是文件夹或文件
        if os.path.isfile(item_path):
            os.remove(item_path)  # 删除文件
            print(f"删除文件: {item_path}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # 删除文件夹及其内容
            print(f"删除文件夹: {item_path}")
    
    print(f"文件夹 '{folder_path}' 已清空")

# ------------------- 模型初始化 ---------------
# --- SenceVoice-语音识别模型
model_dir = r"E:\2_PYTHON\Project\GPT\QWen\pretrained_models\SenseVoiceSmall"
model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True, )

# --- QWen2.5大语言模型 ---
# model_name = r":\2_PYTHON\Project\GPT\QWen\Qwen2.5-0.5B-Instruct"
model_name = r"E:\2_PYTHON\Project\GPT\QWen\Qwen2.5-1.5B-Instruct"
# model_name = r':\2_PYTHON\Project\GPT\QWen\Qwen2.5-7B-Instruct-GPTQ-Int4'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- CosyVoice - 语音合成模型
# CosyVoice-300M和CosyVoice-300M-SFT的区别主要体现在以下几个方面：
# 训练方式
# CosyVoice-300M：是基座模型，未经过特定任务的微调，属于通用模型。
# CosyVoice-300M-SFT：是在CosyVoice-300M的基础上，通过监督式微调（Supervised Fine-Tuning）训练得到的。
# 音色特点
# CosyVoice-300M：擅长准确代表说话者身份，能够通过3至10秒的音频样本克隆出音色。
# CosyVoice-300M-SFT：内置了多个预训练音色，如中文女声、中文男声、日语男声、粤语女声等。
# 以下模型名称从CosyVoice-300M改为CosyVoice-300M-SFT，修复运行后提示不存在中文女的报错，因为CosyVoice-300M中没有内置了多个预训练音色
cosyvoice = CosyVoice(r'E:\2_PYTHON\Project\GPT\QWen\pretrained_models\CosyVoice-300M-SFT', load_jit=True, load_onnx=False, fp16=True)
# --- CosyVoice - 支持的音色列表
print(cosyvoice.list_avaliable_spks())
# ------------------ 模型初始化结束 ----------------

while(1):
    # 使用函数录音，作为输入
    record_audio("my_recording.wav")

    # input_file = ( "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav" )
    input_file = ("my_recording.wav")
    res = model_senceVoice.generate(
        input=input_file,
        cache={},
        language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
    )

    # -------- 模型推理阶段，将语音识别结果作为大模型Prompt ------
    prompt = res[0]['text'].split(">")[-1] + "，回答简短一些，保持50字以内！"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Input:", prompt)
    print("Answer:", response)

    # --- 答复输出文件夹 ---
    folder_path = "./out_answer/"
    clear_folder(folder_path)

    # ['中文女', '中文男', '日语男', '粤语女', '英文女', '英文男', '韩语女']
    # change stream=True for chunk stream inference
    index_out = 0
    for i, j in enumerate(cosyvoice.inference_sft(f'{response}', '中文女', stream=False)):
        torchaudio.save('{}/sft_{}.wav'.format(folder_path,i), j['tts_speech'], 22050)
        index_out += 1
        # play_audio('sft_{}.wav'.format(i))

    for idx in range(index_out):
        play_audio('{}/sft_{}.wav'.format(folder_path,idx))

