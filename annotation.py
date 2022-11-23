# 自动标注文件
import os
import re
import tqdm
import jieba.posseg as psg
import json
import shutil
import uuid
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.tts.infer import TTSExecutor
from paddlespeech.t2s.exps.syn_utils import get_frontend

import random
import string
 

def random_string_generator(str_size):
    allowed_chars = "abcdedghijklmnopqrstuvwxyz"
    return ''.join(random.choice(allowed_chars) for x in range(str_size))

def status_change(status_path, status):
    with open(status_path, "w", encoding="utf8") as f:
        status = {
            'on_status': status
        }
        json.dump(status, f, indent=4)

# 初始化 TTS，自动下载预训练模型
tts = TTSExecutor()
tts(text="今天天气十分不错。", output="output.wav")

# 初始化 ASR，自动下载预训练模型
asr = ASRExecutor()
result = asr(audio_file="output.wav", force_yes=True)


def get_pinyins(sentences):
    # frontend = get_frontend(
    #     lang="mix",
    #     phones_dict="/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0/phone_id_map.txt",
    #     tones_dict=None)
    
    segments = sentences
    for seg in segments:
        # Replace all English words in the sentence
        seg = re.sub('[a-zA-Z]+', '', seg)
        seg_cut = psg.lcut(seg)

        seg_cut = tts.frontend.tone_modifier.pre_merge_for_modify(seg_cut)
        all_pinyins = []
        # 为了多音词获得更好的效果，这里采用整句预测
        if tts.frontend.g2p_model == "g2pW":
            try:
                pinyins = tts.frontend.g2pW_model(seg)[0]
            except Exception:
                # g2pW采用模型采用繁体输入，如果有cover不了的简体词，采用g2pM预测
                print("[%s] not in g2pW dict,use g2pM" % seg)
                pinyins = tts.frontend.g2pM_model(seg, tone=True, char_split=False)
            pre_word_length = 0
            for word, pos in seg_cut:
                now_word_length = pre_word_length + len(word)
                if pos == 'eng':
                    pre_word_length = now_word_length
                    continue
                word_pinyins = pinyins[pre_word_length:now_word_length]
                # 矫正发音
                word_pinyins = tts.frontend.corrector.correct_pronunciation(
                    word, word_pinyins)
                all_pinyins.extend(word_pinyins)
                pre_word_length = now_word_length
    return all_pinyins 


def trans_wav_format(input_path, output_path):
    # 统一输入音频格式
    cmd = f"ffmpeg -i {input_path} -ac 1 -ar 24000 -acodec pcm_s16le {output_path}"
    print(cmd)
    os.system(cmd)
    if not os.path.exists(output_path):
        print(f"文件转换失败，请检查报错: {input_path}")
        return None
    else:
        return output_path

def annotation_dataset(data_dir, label_path, temp_dir="/home/aistudio/temp", use_ffmpeg=True):
    # 先清空 temp 目录
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    ann_json = "/home/aistudio/work/ann_status.json"
    status_change(ann_json, True)

    ann_result = []

    wavs = [filename for filename in os.listdir(data_dir) if filename[-4:] in ['.wav', '.mp3', '.ogg']]
    if len(wavs) < 5:
        print("数据小于5句，不建议微调，请添加数据")
        return
    
    for idx, filename in tqdm.tqdm(enumerate(wavs)):
        # 检查文件名中是否有非法字符，存在非法字符则重新命名
        input_path = os.path.join(data_dir, filename)
        if " " in filename:
            new_filename = str(idx) + "_" + random_string_generator(4) + ".wav"
            filename = new_filename
            new_file_path = os.path.join(data_dir, new_filename)
            os.rename(input_path, new_file_path)
            print(f"文件名不合法：{input_path}")
            print(f"重命名结果：{new_file_path}")
            input_path = new_file_path
        
        if filename[-4:] != ".wav":
            filename = filename[:-4] + ".wav"

        if use_ffmpeg:
            # 使用 ffmpeg 统一音频格式
            output_path = os.path.join(temp_dir, filename)
            output_path = trans_wav_format(input_path, output_path)
            filepath = output_path
        else:
            filepath = input_path

        if filepath:
            asr_result = asr(audio_file=filepath, force_yes=True)
            pinyin = " ".join(get_pinyins([asr_result]))
            ann_result.append(
                {
                    "filename": filename,
                    "filepath": filepath,
                    "asr_result": asr_result,
                    "pinyin": pinyin
                }
            )
    status_change(ann_json, False)
    
    if ann_result:
        ann_result = sorted(ann_result, key=lambda x:x['filename'])
        if label_path:
            with open(label_path, "w", encoding="utf8") as f:
                json.dump(ann_result, f, indent=2, ensure_ascii=False)
    else:
        return None
    return ann_result

