# 安装 PaddleSpeech
if [ ! -d "/home/aistudio/PaddleSpeech" ];then
# PaddleSpeech 不存在
cd /home/aistudio \
&& wget https://paddlespeech.bj.bcebos.com/demos/speech_web/PaddleSpeech.zip \
&& unzip PaddleSpeech.zip \
&& rm PaddleSpeech.zip
else
echo "PaddleSpeech exits"
fi

# 下载 nltk 依赖
if [ ! -d "/home/aistudio/nltk_data" ];then
cd /home/aistudio \
&& wget https://paddlespeech.bj.bcebos.com/Parakeet/tools/nltk_data.tar.gz \
&& tar zxvf nltk_data.tar.gz \
&& rm nltk_data.tar.gz
else
echo "nltk_data exits"
fi

# 删除死链
find -L /home/aistudio -type l -delete

# pip 安装依赖库
cd /home/aistudio/PaddleSpeech \
&& pip install pytest-runner -i https://mirror.baidu.com/pypi/simple \
&& pip install . -i https://mirror.baidu.com/pypi/simple \
&& pip install paddlespeech-ctcdecoders -i https://mirror.baidu.com/pypi/simple \
&& pip install uvicorn==0.18.3 -i https://mirror.baidu.com/pypi/simple

# 下载预训练模型
if [ ! -d "/home/aistudio/.paddlespeech/models" ];then
# 没有安装PaddleSpeech
paddlespeech tts --input "你好，欢迎使用百度飞桨深度学习框架！" --output output.wav \
&& paddlespeech asr --lang zh --input output.wav --yes \
&& rm output.wav
else
echo "paddlespeech 预训练模型已下载"
fi


# 配置 MFA 环境
if [ ! -d "/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/tools" ];then
cd /home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3 \
&& mkdir -p tools/aligner \
&& cd tools \
&& cp /home/aistudio/montreal-forced-aligner_linux.tar.gz ./ \
&& tar xvf montreal-forced-aligner_linux.tar.gz \
&& cd montreal-forced-aligner/lib \
&& ln -snf libpython3.6m.so.1.0 libpython3.6m.so \
&& cd /home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/tools/aligner \
&& wget https://paddlespeech.bj.bcebos.com/MFA/ernie_sat/aishell3_model.zip \
&& wget https://paddlespeech.bj.bcebos.com/MFA/AISHELL-3/with_tone/simple.lexicon \
&& unzip aishell3_model.zip
else
echo "MFA 环境已存在"
fi

# 下载微调需要的训练模型
if [ ! -d "/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/models" ];then
cd /home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3 \
&& mkdir models \
&& cd models \
&& wget https://paddlespeech.bj.bcebos.com/t2s/chinse_english_mixed/models/fastspeech2_mix_ckpt_1.2.0.zip \
&& unzip fastspeech2_mix_ckpt_1.2.0.zip \
&& wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwgan_aishell3_static_1.1.0.zip \
&& unzip pwgan_aishell3_static_1.1.0.zip \
&& wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_aishell3_static_1.1.0.zip \
&& unzip hifigan_aishell3_static_1.1.0.zip \
&& wget https://paddlespeech.bj.bcebos.com/Parakeet/released_models/wavernn/wavernn_csmsc_static_0.2.0.zip \
&& unzip wavernn_csmsc_static_0.2.0.zip
else
echo "预训练模型下载完成"
fi