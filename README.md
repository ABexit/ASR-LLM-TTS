# 241123， UPDATE 😄

单模态语音交互
13_SenceVoice_QWen2.5_edgeTTS_realTime.py

音视频多模态语音交互
14_SenceVoice_QWen2VL_edgeTTS_realTime.py


# 语音交互大模型/SenceVoice-QWen2.5-TTS

## 框架
SenceVoice-QWen2.5-CosyVoice搭建

此工程主代码来于[CosyVoice] (https://github.com/FunAudioLLM/CosyVoice)

在CosyVoice基础上添加[SenceVoice] (https://github.com/modelscope/FunASR) 作为语音识别模型

添加[QWwn2.5] (https://github.com/QwenLM/Qwen2.5) 作为大语言模型进行对话理解

## 3种语音合成方法

CoosyVoice推理速度慢，严重影响对话实时性，额外添加pyttsx3和edgeTTS

EdgeTTS实验过程出现链接错误问题，升级版本至6.1.17解决，无需科学上网

All dependencies are listed in requirements.txt, the interactive inference scripts are 10/11/12_SenceVoice_QWen2.5_xxx.py. 

Have fun! 😊
