#该应用创建工具共包含三个区域，顶部工具栏，左侧代码区，右侧交互效果区，其中右侧交互效果是通过左侧代码生成的，存在对照关系。
#顶部工具栏：运行、保存、新开浏览器打开、实时预览开关，针对运行和在浏览器打开选项进行重要说明：
#[运行]：交互效果并非实时更新，代码变更后，需点击运行按钮获得最新交互效果。
#[在浏览器打开]：新建页面查看交互效果。
#以下为应用创建工具的示例代码

import streamlit as st
import time
import psutil

st.header("使用自己的声音做语音合成")
"通过这个应用，你可以上传自己的数据，使用自己想要的音色做语音合成工作，希望你在这个项目玩得开心！"

# pl = psutil.pids()
# for pid in pl:
#     try:
#         name = psutil.Process(pid).name()
#         pid = pid
#         st.write(f"pid: {pid}  name:{name}")
#     except:
#         continue


import os
import json
from annotation import annotation_dataset
from finetuneTTS import finetuneTTS, generateTTS, find_max_ckpt, exp_base, inference_dir,finetune_train, status_change
import multiprocessing
import psutil
import numpy as np

os.makedirs(inference_dir, exist_ok=True)

def clear_mem():
    # 有时候会产生 lsof 抢占资源的情况，页面刷新时自动清除
    cmd = "ps -ef | grep lsof | grep -v grep | awk '{print $2}' | xargs kill -9"
    os.system(cmd)

def clear_data():
    cmd = "rm -rf work/data/*"
    os.system(cmd)
    cmd = "rm -rf /home/aistudio/work/labels_data.json"
    os.system(cmd)

def clear_label():
    cmd = "rm -rf /home/aistudio/work/labels_data.json"
    os.system(cmd)

clear_mem()

##############################
## 步骤0： 可视化界面默认配置参数 #
##############################
data_dir = "/home/aistudio/work/data"
label_path = "/home/aistudio/work/labels_data.json"
exp_name_path = "/home/aistudio/work/exp.txt"
os.makedirs(data_dir, exist_ok=True)


#####################
## 步骤1： 上传数据集 #
####################

st.markdown("<hr />",unsafe_allow_html=True)
st.write("**1、上传数据**")

st.markdown(
    """

**注意事项**

**建议第一次使用时不要上传太多的音频，先用小数据集，少的训练轮次跑一遍，确定环境流程没有问题！一旦参数调大，中间会等待的时间比较长！**

> 上传数据说明：
> 
> 对于语音合成任务，对数据是有一定要求的，尽可能上传干净的人声数据，比如像示例中的人声数据，在安静环境下录制，录制设备无论是手机，电脑，还是别的设备都可以，注意一定要控制噪音，或者提前使用音频剪辑软件进行降噪。（一定要是中文数据，这个项目目前只支持中文数据！！其它语言预标注环节会出错！）
> 
> 1. 音频不要太长，也不要太短，2s~10s之间，音频太长会报错！
> 2. 音频尽量是干净人声，不要有BGM，不要有比较大的杂音，不要有一些奇奇怪怪的声效，比如回声等
> 3. 声音的情绪尽量稳定,以说话的语料为主，不要是『嗯』『啊』『哈』之类的语气词
> 4. 音频数量大于5 条！！否则会报错！！
> 
> 关于录音工具的选择：
> 
> 你可以使用一些在线运行的录音工具或者 【Adobe Audition】，【Cool Edit Pro】, 【Audacity】 等录音软件录制音频，保存为 24000采样率的 Wav 格式
> 
> 也可以通过手机录音后，使用格式工厂等格式转换软件，将手机录音转换成 Wav 格式后上传到这个项目中。
> 
> 希望大家玩得开心！！

**示例音频下载**

+ 示例发音人1： [【SpkA】](https://paddlespeech.bj.bcebos.com/datasets/Aistudio/finetune/SpkA.zip)
+ 示例发音人2：[【SpkB】](https://paddlespeech.bj.bcebos.com/datasets/Aistudio/finetune/SpkB.zip)

解压后可将 Wav 文件上传

**关于如何制作数据集**

可以参考这篇 [项目游玩注意事项和数据集制作经验](https://www.bilibili.com/read/cv19722919?spm_id_from=333.999.0.0)

"""
)

st.write("**点击【Browse files】上传数据集，单次上传太多会卡，数据太多可分次上传**")

uploaded_files = st.file_uploader("上传音频", type=['wav', 'ogg', 'mp3'], label_visibility='hidden', accept_multiple_files=True, on_change=clear_label)

if uploaded_files is not None:
    # To read file as bytes:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.getvalue()
        filename = uploaded_file.name
        with open(os.path.join(data_dir, filename), "wb") as f:
            f.write(bytes_data)

max_show = 50

st.write(f"已上传音频文件（最多展示{max_show}条，剩余的不做展示）")
files = []
markdown_str = """
| f1 | f2 | f3 | f4 | f5 |
| :----: | :----: | :----: | :----: | :----: |
"""
cnt = 0

for idx, filename in enumerate(sorted(os.listdir(data_dir))):
    if filename.endswith(".wav") or filename.endswith(".mp3") or filename.endswith(".ogg"):
        files.append(filename)
        cnt += 1
        if cnt <= max_show:
            if idx % 5 == 0:
                markdown_str += f"| {filename} |"
            elif idx % 5 == 4:
                markdown_str += f" {filename} |\n"
            else:
                markdown_str += f" {filename} |"
st.markdown(markdown_str)
st.write(f"音频共计上传 【{cnt}】 条")


# 如果你需要重新
st.write("如果上传的数据有问题，你可以点击【清除数据】把原来的数据清空，再次上传时记得【刷新一下网页】")
st.button("清除上传数据", on_click=clear_data)

#####################
## 步骤2： 预标注数据 #
####################
if 'label_result' not in st.session_state:
    st.session_state.label_result = False

def ann_labels(data_dir, label_path):
    if os.path.exists(label_path):
        os.remove(label_path)
    st.session_state.label_result = annotation_dataset(data_dir, label_path)


st.markdown("<hr />",unsafe_allow_html=True)
st.write("**2. 数据预标注 & 检验**")
st.markdown(
    "点击【检验数据】按钮，开始对数据进行预标注与检验，等出现【数据检验成功】字样以后，再执行下一步"
)
st.button("检验数据", on_click=ann_labels, args=(data_dir, label_path))

label_check = st.empty()

# st.write(st.session_state.label_result)
if os.path.exists(label_path):
    st.session_state.label_result = True
else:
    ann_json = "/home/aistudio/work/ann_status.json"
    if os.path.exists(ann_json):
        with open(ann_json, "r", encoding="utf8") as f:
            ft_status_dict = json.load(f)
        if ft_status_dict['on_status']:
            label_check.warning('文件还在标注中，请耐心等待！')
        else:
            if os.path.exists("/home/aistudio/work/labels_data.json"):
                label_check.success("文件标注完成")

if os.path.exists("/home/aistudio/work/labels_data.json"):
    label_check.success("数据检验成功!请执行下一步！")
elif st.session_state.label_result is None:
    label_check.error("数据检验失败！！！请检查数据是否上传输入正确！检查文件是否为指定格式，音频长度是否符合要求，是否存在过长或者过短的音频!!这一步有问题就不要执行下一步了！！先检查数据是否符合要求！！")


######################
## 步骤3： 微调模型 ####
#####################
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**3. 选择适合的参数微调数据**")

st.markdown("""

不同的训练轮次，学习音色的效果不同，你可以多做几轮试验进行尝试！训练是增量训练，比如先训练了100个轮次，再训练100个轮次，则最终的结果会被训练200个轮次。你可以根据自己的试验结果做调整！

一般来说，训练轮次越多，效果会更好一些，但是训练的时间也会更长，训练时注意保持网络链接，断网一段时间的话进程会被停掉。如果你的数据比较少，可以适当增大训练轮次；如果你的数据比较多，可以适当减少训练轮次

这里给出几组时间参考，

| 数据量 | 训练轮次 | 耗时 |
| :----: | :----: | :----: |
| 12句 | 100 | 5分钟左右 |
| 12句 | 500 | 20分钟左右 |
| 3000句 | 100 | 180分钟左右 |


"""
,unsafe_allow_html=True)


if 'finetune_result' not in st.session_state:
    st.session_state.finetune_result = False

# 微调中
if 'finetune_on' not in st.session_state:
    st.session_state.finetune_on = False

def ft_st(label_path, exp_name, epoch):
    st.session_state.finetune_result = False
    st.write("开始微调训练", "微调中请耐心等待！")
    st.session_state.finetune_on = True

st.markdown(
    """点击【微调训练】按钮，开始对数据进行微调，在这个过程中请耐心等待！直至本步骤下方出现【恭喜你！微调训练成功！！开始使用自己的声音做语音合成吧！！】再进行下一步，如果未训练成功，请再次检查输入的数据是否符合要求！
    开始训练后20s左右出现进度条，请耐心等待，刚开始实验时建议先用小数据集验证数据可行性！得到一个初步的数据效果过，然后再调参，加大数据或者增加训练的步数。

    等训练成功有模型出现之后，再点击【导出模型】，导出模型后才会出现在实验模型选择界面和合成按键。

    > 如果【训练步数】小于模型已训练步数，则不会进行训练，当【训练步数】大于模型已训练步数，则会增量训练。举例：
    下方提示已存在模型 【snapshot_iter_200.pdz】，表示我们之前已经训练了一个 200 步的模型了
    1. 如果我们输入训练步数为 100 , 则不会进行训练，模型已经学了200步了，已经学够了
    2. 如果我们输入训练步数为 300 , 则会在200步的这个模型上继续训练
    """
)

epoch = 100
epoch = st.number_input('请输入训练步数，如 100, 200, 300, 最小100, 最大20000', min_value=100, max_value=20000, step=100)

if os.path.exists(exp_name_path):
    with open(exp_name_path, "r", encoding="utf8") as f:
        exp_name = f.readlines()[0].strip()
    exp_name = st.text_input("请输入实验名称", value=exp_name)
else:
    exp_name = st.text_input("请输入实验名称", value="demo")

with open(exp_name_path, "w", encoding="utf8") as f:
        f.write(exp_name)
st.button("微调训练", on_click=ft_st, args=(label_path, exp_name, epoch))

label_finetune = st.empty()
label_checkpoint = st.empty()

exp_path = exp_base + exp_name
model_path = os.path.join(exp_path, "output/checkpoints")

my_bar = st.progress(0)
label_check_status = st.empty()

def finetune_train_sub(dump_dir, output_dir, max_step):
    cmd = f"""python finetuneTrain.py --dump_dir {dump_dir} --output_dir {output_dir} --max_step {max_step}
    """
    os.system(cmd)

exp_path = exp_base + exp_name
dump_dir = os.path.join(exp_path, "dump")
output_dir = os.path.join(exp_path, "output")
train_status_json = os.path.join(output_dir, "train_status.json")
from multiprocessing import Process

# 当被点击微调训练以后
if st.session_state.finetune_on:
    label_finetune.warning("开始微调训练，请耐心等待！(20s以后出现进度条)")
    # 先执行数据预处理
    finetuneTTS(label_path, exp_name,start_step=1, end_step=6, epoch=epoch)
    # 再执行微调
    # 先检查有没有子进程，没有子进程再运行
    process_name = "finetune_train"

    max_step = epoch
    train_status = False
    if os.path.exists(train_status_json):
        train_status = True

    if not train_status:
        # 没有子进程在跑
        # 开子进程微调训练
        p = Process(target=finetune_train_sub, args=(dump_dir, output_dir, max_step), name=process_name)
        p.start()
        time.sleep(15)

#  看序列化保存的文件中是否开始训练
ft_status_json = os.path.join(exp_path, "finetune_status.json")
if os.path.exists(ft_status_json):
    with open(ft_status_json, "r", encoding="utf8") as f:
        ft_status_dict = json.load(f)
    if ft_status_dict['on_status']:
        # 生成进度条
        if os.path.exists(train_status_json):
            # 这个为 True 但是没有训练中间文件
            label_finetune.warning('微调训练中！耗时较长，请耐心等待！(20s以后出现进度条)')
            with open(train_status_json, "r", encoding="utf8") as f:
                status = json.load(f)
            percent = int(status['step'] / status['max_step'] * 100)
            error = 0
            while percent < 100:
                time.sleep(1)
                if not os.path.exists(train_status_json):
                    break
                else:
                    try:
                        error = 0
                        with open(train_status_json, "r", encoding="utf8") as f:
                            status = json.load(f)
                        percent = int(status['step'] / status['max_step'] * 100)
                        my_bar.progress(percent)
                        label_check_status.success(f"step: {status['step']}  max_step:{status['max_step']} loss: {status['loss']} 进度：{percent}%")
                    except:
                        error += 1
                        if error > 10:
                            break
            # 训练结束后删除
            cmd = f"rm {train_status_json}"
            os.system(cmd)
            label_finetune.success("364 恭喜你！微调训练成功！！开始使用自己的声音做语音合成吧！！")
        else:
            # 纠错
            status_change(ft_status_json, False)
    else:
        # finetune status 为 False 且模型存在
        if os.path.exists(model_path):
            ckpt = find_max_ckpt(model_path)
            ckpt_model_path = f"{model_path}/snapshot_iter_{ckpt}.pdz"
            if os.path.exists(ckpt_model_path):
                if not os.path.exists(train_status_json):
                    label_finetune.success("371 恭喜你！微调训练成功！！开始使用自己的声音做语音合成吧！！")
                st.session_state.finetune_on = False
            else:
                label_finetune.error("未发现模型，请重新微调")

# 再看模型是否存在
if os.path.exists(model_path):
    ckpt = find_max_ckpt(model_path)
    if ckpt != 99200:
        # 默认预训练模型
        ckpt_model_path = f"{model_path}/snapshot_iter_{ckpt}.pdz"

        if os.path.exists(ckpt_model_path):
            st.write(f"已存在模型：{ckpt_model_path}")

def export_model_step8(label_path, exp_name, epoch):
    finetuneTTS(label_path, exp_name,start_step=8, end_step=8, epoch=epoch)

st.write("当上方提示已存在模型 *** 之后(要是进度条到了100%之后要是没出现，就手动刷新一下按F5)，手动导出静态图模型到inference目录下，导出成功后可以进行合成，当有新的模型训练产生后，点击导出模型可以覆盖之前训练的旧模型")

st.button("导出模型", on_click=export_model_step8, args=(label_path, exp_name, epoch))
label_export_static = st.empty()
if os.path.exists(os.path.join(inference_dir, exp_name)):
    label_export_static.success(f"导出静态图模型成功")


################################
## 步骤4： 使用自己的声音做合成 ####
###############################
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**4. 使用你自己的声音做合成**")

st.markdown("""
声码器说明：这里预制了三种声码器【PWGan】【WaveRnn】【HifiGan】, 三种声码器效果和生成时间有比较大的差距，请跟进自己的需要进行选择。

| 声码器 | 音频质量 | 生成速度 |
| :----: | :----: | :----: |
| PWGan | 中等 | 中等 |
| WaveRnn | 高 | 非常慢（耐心等待） |
| HifiGan | 低 | 快 |

""")

if 'generate_result' not in st.session_state:
    st.session_state.generate_result = False

# 合成中
if 'generate_on' not in st.session_state:
    st.session_state.generate_on = False

def generate_(wav_output_dir):
    if os.path.exists(os.path.join(wav_output_dir, "1.wav")):
        cmd = f"rm {os.path.join(wav_output_dir, '1.wav')}"
        os.system(cmd)
    st.session_state.generate_on = True

text = st.text_input("输入文本，支持中英双语！", value="欢迎使用 Paddle Speech 做语音合成工作！一起来玩吧！")

voc = st.selectbox(
    '选择声码器',
    ('PWGan', 'WaveRnn', 'HifiGan'))

if os.path.exists("inference"):
    exp_list = [dirname for dirname in os.listdir("inference") if os.path.isdir(f"inference/{dirname}") and ".ipynb_" not in dirname]

    if len(exp_list) > 0:
        gen_exp_name = st.selectbox(
            '选择实验模型',
            tuple(exp_list),
            )

        text_dict = {
            "1": text
        }
        wav_output_dir = os.path.join(exp_base+gen_exp_name, "wav_out")
        st.button("合成", on_click=generate_, args=(wav_output_dir,))

        label = st.empty()

        if st.session_state.generate_on:
            label.warning('音频合成中，请耐心等待')
            st.session_state.generate_result = generateTTS(text_dict, gen_exp_name, voc)

        if st.session_state.generate_result is None:
            st.write("音频合成失败！！请检查报错信息，如果实在不知道怎么调试就把报错信息截图发在交流群里面吧！")

        status_json = os.path.join(exp_base+gen_exp_name, "generate_status.json")
        if os.path.exists(status_json):
            with open(status_json, "r", encoding="utf8") as f:
                gen_status_dict = json.load(f)
            if gen_status_dict['on_status']:
                label.warning('音频合成中，请耐心等待！')
            else:
                label.success('音频合成成功！')
                st.session_state.generate_on = False

        if os.path.exists(os.path.join(wav_output_dir, "1.wav")):
            st.audio(os.path.join(wav_output_dir, "1.wav"))
            st.session_state.generate_on = False

###############################
## 步骤5： 模型导出 ###########
###############################
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**5. 模型导出**")
st.write("生成对应实验的zip文件，方便大家下载，如果大家拿到其他的模型压缩文件，可以放在 inference 目录下解压，在第四步【选择实验模型】的选项中，可以选择其他人的模型")


if len(exp_list) > 0:
    export_exp_name = st.selectbox(
        '选择想要导出的实验',
        tuple(exp_list),
        )
    label_export = st.empty()
    export_status = False
    
    def export_model(export_exp_name, label_export):
        # 检查里面是否有文件
        file_list = ["fastspeech2_mix.pdiparams", "fastspeech2_mix.pdiparams.info", "fastspeech2_mix.pdmodel", "phone_id_map.txt"]
        exp_inference_dir = os.path.join(inference_dir, export_exp_name)
        for file_use in file_list:
            if not os.path.exists(os.path.join(exp_inference_dir, file_use)):
                label_export.error("文件不全，请检查上方模型训练是否结束")
                return False

        if os.path.exists(exp_inference_dir+".zip"):
            cmd = f"rm {exp_inference_dir}.zip"
            os.system(cmd)

        cmd = f"cd {inference_dir} && zip -r {export_exp_name}.zip {export_exp_name}/"
        res = os.system(cmd)
        export_status = True    

    st.button("打包模型文件", on_click=export_model, args=(export_exp_name, label_export))
    if export_status:
        label_export.success(f"模型打包成功：{inference_dir}/{export_exp_name}.zip ，可前往对应路径下载")


###############################
## 步骤6： 模型下载与上传 ###########
###############################
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**6. 模型下载与上传**")

st.write("""
回到 BML CodeLab 主页，进入 inference 目录下，在这个目录下进行模型的下载与上传，都是 ".zip" 文件，上传完成后在这个目录下解压即可使用！
""")

################################
## 步骤7： 一键重置 ####
###############################
st.markdown("<hr />",unsafe_allow_html=True)
st.write("**7. 更换发音人，重新跑试验**")

st.markdown("""
执行【一键重置】会删除上面已经训练的模型，如果你想重新训练一个新的发音人模型的话，你可以选择【一键重置】
""")

def restart_exp(exp_name):
    cmd = f"cd /home/aistudio/work/ && rm -rf exp_* && rm labels_data.json && rm *_status.json"
    os.system(cmd)

st.button("一键重置", on_click=restart_exp, args=(exp_name, ))
