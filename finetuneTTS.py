import os
import shutil
import subprocess
import yaml
from yacs.config import CfgNode
import json
from pathlib import Path
import soundfile as sf

from paddlespeech.t2s.exps.syn_utils import am_to_static
from paddlespeech.t2s.exps.syn_utils import get_am_inference
from paddlespeech.t2s.exps.syn_utils import get_am_output
from paddlespeech.t2s.exps.syn_utils import get_frontend
from paddlespeech.t2s.exps.syn_utils import get_predictor
from paddlespeech.t2s.exps.syn_utils import get_voc_output

from typing import List

import jsonlines
import numpy as np
import paddle
from paddle.io import DataLoader
from paddle.io import DistributedBatchSampler
from paddlespeech.t2s.datasets.am_batch_fn import fastspeech2_multi_spk_batch_fn
from paddlespeech.t2s.datasets.data_table import DataTable
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2
from paddlespeech.t2s.training.optimizer import build_optimizers
from paddlespeech.t2s.models.fastspeech2 import FastSpeech2Loss

# 配置路径信息
# 这里的路径是提前写好在 Aistudio 上的路径信息，如果需要迁移到其它的环境，请自行配置
cwd_path = "/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3"
pretrained_model_dir = "/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/models/fastspeech2_mix_ckpt_1.2.0"
config_path = "/home/aistudio/PaddleSpeech/examples/other/tts_finetune/tts3/conf/finetune.yaml"
inference_dir = "/home/aistudio/inference"
exp_base = "/home/aistudio/work/exp_"

os.makedirs(inference_dir, exist_ok=True)

pwg_inference_dir = os.path.join(cwd_path, "models/pwgan_aishell3_static_1.1.0")
hifigan_inference_dir = os.path.join(cwd_path, "models/hifigan_aishell3_static_1.1.0")
wavernn_inference_dir = os.path.join(cwd_path, "models/wavernn_csmsc_static_0.2.0")


# 命令行执行函数，可以进入指定路径下执行
def run_cmd(cmd, cwd_path, message, check_result_path=None):
    p = subprocess.Popen(cmd, shell=True, cwd=cwd_path)
    res = p.wait()
    print(cmd)
    print("运行结果：", res)

    # 检查是否有对应文件生成
    if check_result_path:
        file_num = [filename for filename in os.listdir(check_result_path)]
        if file_num == 0:
            res = 1


    if res == 0:
        # 运行成功
        return True
    else:
        # 运行失败
        print(message)
        print(f"你可以新建终端，打开命令行，进入：{cwd_path}")
        print(f"执行：{cmd}")
        print("查看命令行中详细的报错信息")
        return False

def status_change(status_path, status):
    with open(status_path, "w", encoding="utf8") as f:
        status = {
            'on_status': status
        }
        json.dump(status, f, indent=4)

# 找到最新生成的模型
def find_max_ckpt(model_path):
    max_ckpt = 0
    for filename in os.listdir(model_path):
        if filename.endswith('.pdz'):
            files = filename[:-4]
            a1, a2, it = files.split("_")
            if int(it) > max_ckpt:
                max_ckpt = int(it)
    return max_ckpt

# Step1 : 生成标准数据集
def step1_generate_standard_dataset(label_path, exp_path, data_dir):
    # try:
    print("Step1 开始执行: 生成标注数据集")
    # 生成 data 数据路径
    exp_data_path = data_dir
    os.makedirs(exp_data_path, exist_ok=True)
    exp_data_label_file = os.path.join(exp_data_path, "labels.txt")

    with open(label_path, "r", encoding='utf8') as f:
        ann_result = json.load(f)

    # 复制音频文件
    for label in ann_result:
        shutil.copy(label['filepath'], exp_data_path)

    # 生成标准标注文件
    with open(exp_data_label_file, "w", encoding="utf8") as f:
        for ann in ann_result:
            f.write(f"{ann['filename'].split('.')[0]}|{ann['pinyin']}\n")

    return exp_data_path
    # except:
    #     print(traceback.print_exc())
    #     print("Step1 生成标注数据集 执行失败，请检查数据集是否配置正确")
    #     return None
    

# Step2: 检查非法数据
def step2_check_oov(data_dir, new_dir, lang="zh"):
    print("Step2 开始执行: 检查数据集是否合法")
    cmd = f"""
            python3 local/check_oov.py \
                --input_dir={data_dir} \
                --pretrained_model_dir={pretrained_model_dir} \
                --newdir_name={new_dir} \
                --lang={lang}
            """
    if not run_cmd(cmd, cwd_path, message="Step2 检查非法数据 执行失败，请检查数据集是否配置正确"):
        return False
    else:
        return True

# Step3: 生成 MFA 对齐结果
def step3_get_mfa(new_dir, mfa_dir, lang="zh"):
    print("Step3 开始执行: 使用 MFA 对数据进行对齐")
    cmd = f"""
        python3 local/get_mfa_result.py \
            --input_dir={new_dir} \
            --mfa_dir={mfa_dir} \
            --lang={lang}
        """
    if not run_cmd(cmd, cwd_path, message="Step3 MFA对齐 执行失败，请执行命令行，查看 MFA 详细报错", check_result_path=mfa_dir):
        return False
    else:
        return True

# Step4: 生成 Duration 
def step4_duration(mfa_dir):
    print("Step4 开始执行: 生成 Duration 文件")
    cmd = f"""
        python3 local/generate_duration.py \
            --mfa_dir={mfa_dir}
        """
    if not run_cmd(cmd, cwd_path, message="Step4 生成 Duration 执行失败，请执行命令行"):
        return False
    else:
        return True

# Step5: 数据预处理
def step5_extract_feature(data_dir, dump_dir):
    print("Step5 开始执行: 开始数据预处理")
    cmd = f"""
        python3 local/extract_feature.py \
            --duration_file="./durations.txt" \
            --input_dir={data_dir} \
            --dump_dir={dump_dir}\
            --pretrained_model_dir={pretrained_model_dir}\
        """
    if not run_cmd(cmd, cwd_path, message="Step5 执行失败，请执行命令行"):
        return False
    else:
        return True

# Step6: 准备微调环境
def step6_prepare_env(output_dir):
    print("Step6 开始执行: 准备微调环境")
    cmd = f"""
        python3 local/prepare_env.py \
            --pretrained_model_dir={pretrained_model_dir} \
            --output_dir={output_dir}
    """
    if not run_cmd(cmd, cwd_path, message="Step6 准备训练环境 执行失败，请执行命令行"):
        return False
    else:
        return True

# Step7: 微调训练
def step7_finetune(dump_dir, output_dir, epoch, batch_size=None, learning_rate=None, num_snapshots=1):
    # 读取默认的yaml文件
    with open(config_path) as f:
        finetune_config = yaml.safe_load(f)
    # 多少个 step 保存一次
    finetune_config['num_snapshots'] = num_snapshots
    # 1. 自动调整 batch ，通过 dump/train里面文件数量判断
    if not batch_size:
        train_data_dir = os.path.join(dump_dir, "train/norm/data_speech")
        file_num = len([filename for filename in os.listdir(train_data_dir) if filename.endswith(".npy")])
        if file_num <= 32:
            batch_size = file_num
            finetune_config['batch_size'] = batch_size
        else:
            finetune_config['batch_size'] = 32
    else:
        finetune_config['batch_size'] = batch_size

    # 2. 支持调整 learning_rate
    if learning_rate:
        finetune_config['learning_rate'] = learning_rate

    # 重新生成这次试验需要的yaml文件
    new_config_path = os.path.join(dump_dir, "finetune.yaml")
    with open(new_config_path, "w", encoding="utf8") as f:
        yaml.dump(finetune_config, f)

    print("Step7 开始执行: 微调试验开始")
    cmd = f"""
        python3 local/finetune.py \
            --pretrained_model_dir={pretrained_model_dir} \
            --dump_dir={dump_dir} \
            --output_dir={output_dir} \
            --ngpu=1 \
            --epoch={epoch} \
            --finetune_config={new_config_path}
        """
    if not run_cmd(cmd, cwd_path, message="Step7 微调试验失败 执行失败，请执行命令行"):
        return False
    else:
        return True

# 导出成静态图
def step8_export_static_model(exp_name):
    exp_path = os.path.join(exp_base + exp_name)
    model_path = os.path.join(exp_path, "output/checkpoints")
    dump_dir = os.path.join(exp_path, "dump")
    output_dir = os.path.join(exp_path, "output")
    ckpt = find_max_ckpt(model_path)
    
    am_config = f"{cwd_path}/models/fastspeech2_mix_ckpt_1.2.0/default.yaml"

    with open(am_config) as f:
        am_config = CfgNode(yaml.safe_load(f))
    
    am_inference = get_am_inference(
        am="fastspeech2_mix",
        am_config=am_config,
        am_ckpt=f"{output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz",
        am_stat=f"{cwd_path}/models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy",
        phones_dict=f"{dump_dir}/phone_id_map.txt",
        tones_dict=None,
        speaker_dict=f"{dump_dir}/speaker_id_map.txt")

    out_inference_dir = os.path.join(inference_dir, exp_name)

    am_inference = am_to_static(
            am_inference=am_inference,
            am="fastspeech2_mix",
            inference_dir=out_inference_dir,
            speaker_dict=f"{dump_dir}/speaker_id_map.txt")
    # 把 phone_dict 也复制过去
    shutil.copy(f"{dump_dir}/phone_id_map.txt", out_inference_dir)
    


# 使用微调后的模型生成音频
def step9_generateTTS(text_dict, wav_output_dir, exp_name, voc="PWGan"):
    # 配置一下参数信息
    exp_path = os.path.join(exp_base + exp_name)
    model_path = os.path.join(exp_path, "output/checkpoints")
    output_dir = os.path.join(exp_path, "output")
    dump_dir = os.path.join(exp_path, "dump")
    
    text_file = os.path.join(exp_path, "sentence.txt")
    status_json = os.path.join(exp_path, "generate_status.json")
    # 写一个物理锁
    status_change(status_json, True)
    
    with open(text_file, "w", encoding="utf8") as f:
        for k,v in sorted(text_dict.items(), key=lambda x:x[0]):
            f.write(f"{k} {v}\n")

    ckpt = find_max_ckpt(model_path)

    if voc == "PWGan":
        cmd = f"""
        python3 /home/aistudio/PaddleSpeech/paddlespeech/t2s/exps/fastspeech2/../synthesize_e2e.py \
                        --am=fastspeech2_mix \
                        --am_config=models/fastspeech2_mix_ckpt_1.2.0/default.yaml \
                        --am_ckpt={output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz \
                        --am_stat=models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy \
                        --voc="pwgan_aishell3" \
                        --voc_config=models/pwg_aishell3_ckpt_0.5/default.yaml \
                        --voc_ckpt=models/pwg_aishell3_ckpt_0.5/snapshot_iter_1000000.pdz \
                        --voc_stat=models/pwg_aishell3_ckpt_0.5/feats_stats.npy \
                        --lang=mix \
                        --text={text_file} \
                        --output_dir={wav_output_dir} \
                        --phones_dict={dump_dir}/phone_id_map.txt \
                        --speaker_dict={dump_dir}/speaker_id_map.txt \
                        --spk_id=0 \
                        --inference_dir
                        --ngpu=1
        """
    elif voc == "WaveRnn":
        cmd = f"""
        python3 /home/aistudio/PaddleSpeech/paddlespeech/t2s/exps/fastspeech2/../synthesize_e2e.py \
                        --am=fastspeech2_mix \
                        --am_config=models/fastspeech2_mix_ckpt_1.2.0/default.yaml \
                        --am_ckpt={output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz \
                        --am_stat=models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy \
                        --voc="wavernn_csmsc" \
                        --voc_config=models/wavernn_csmsc_ckpt_0.2.0/default.yaml \
                        --voc_ckpt=models/wavernn_csmsc_ckpt_0.2.0/snapshot_iter_400000.pdz \
                        --voc_stat=models/wavernn_csmsc_ckpt_0.2.0/feats_stats.npy \
                        --lang=mix \
                        --text={text_file} \
                        --output_dir={wav_output_dir} \
                        --phones_dict={dump_dir}/phone_id_map.txt \
                        --speaker_dict={dump_dir}/speaker_id_map.txt \
                        --spk_id=0 \
                        --ngpu=1
        """
    elif voc == "HifiGan":
        cmd = f"""
        python3 /home/aistudio/PaddleSpeech/paddlespeech/t2s/exps/fastspeech2/../synthesize_e2e.py \
                        --am=fastspeech2_mix \
                        --am_config=models/fastspeech2_mix_ckpt_1.2.0/default.yaml \
                        --am_ckpt={output_dir}/checkpoints/snapshot_iter_{ckpt}.pdz \
                        --am_stat=models/fastspeech2_mix_ckpt_1.2.0/speech_stats.npy \
                        --voc="hifigan_aishell3" \
                        --voc_config=models/hifigan_aishell3_ckpt_0.2.0/default.yaml \
                        --voc_ckpt=models/hifigan_aishell3_ckpt_0.2.0/snapshot_iter_2500000.pdz \
                        --voc_stat=models/hifigan_aishell3_ckpt_0.2.0/feats_stats.npy \
                        --lang=mix \
                        --text={text_file} \
                        --output_dir={wav_output_dir} \
                        --phones_dict={dump_dir}/phone_id_map.txt \
                        --speaker_dict={dump_dir}/speaker_id_map.txt \
                        --spk_id=0 \
                        --ngpu=1
        """
    else:
        print("声码器不符合要求，请重新选择")
        status_change(status_json, False)
        return False
    
    if not run_cmd(cmd, cwd_path, message="Step9 生成音频 执行失败，请执行命令行"):
        status_change(status_json, False)
        return False
    else:
        status_change(status_json, False)
        return True


# 使用静态图生成
def step9_generateTTS_inference(text_dict, exp_name, voc="PWGan"):
    exp_path = exp_base + exp_name
    wav_output_dir = os.path.join(exp_path, "wav_out")
    am_inference_dir = os.path.join(inference_dir, exp_name)
    
    device = "gpu"
    
    status_json = os.path.join(exp_path, "generate_status.json")
    # 写一个物理锁
    status_change(status_json, True)

    # frontend
    frontend = get_frontend(
        lang="mix",
        phones_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
        tones_dict=None
    )

    # am_predictor
    am_predictor = get_predictor(
        model_dir=am_inference_dir,
        model_file="fastspeech2_mix" + ".pdmodel",
        params_file="fastspeech2_mix" + ".pdiparams",
        device=device)
    
    # voc_predictor
    if voc == "PWGan":
        voc_predictor = get_predictor(
            model_dir=pwg_inference_dir,
            model_file="pwgan_aishell3" + ".pdmodel",
            params_file="pwgan_aishell3" + ".pdiparams",
            device=device)
    elif voc == "WaveRnn":
        voc_predictor = get_predictor(
            model_dir=wavernn_inference_dir,
            model_file="wavernn_csmsc" + ".pdmodel",
            params_file="wavernn_csmsc" + ".pdiparams",
            device=device)
    elif voc == "HifiGan":
        voc_predictor = get_predictor(
            model_dir=hifigan_inference_dir,
            model_file="hifigan_aishell3" + ".pdmodel",
            params_file="hifigan_aishell3" + ".pdiparams",
            device=device)

    output_dir = Path(wav_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = list(text_dict.items())

    merge_sentences = True
    fs = 24000
    for utt_id, sentence in sentences:
        am_output_data = get_am_output(
            input=sentence,
            am_predictor=am_predictor,
            am="fastspeech2_mix",
            frontend=frontend,
            lang="mix",
            merge_sentences=merge_sentences,
            speaker_dict=os.path.join(am_inference_dir, "phone_id_map.txt"),
            spk_id=0, )
        wav = get_voc_output(
                voc_predictor=voc_predictor, input=am_output_data)
        # 保存文件
        sf.write(output_dir / (utt_id + ".wav"), wav, samplerate=fs)
    
    status_change(status_json, False)
    return True

def freeze_layer(model, layers: List[str]):
    """freeze layers

    Args:
        layers (List[str]): frozen layers
    """
    for layer in layers:
        for param in eval("model." + layer + ".parameters()"):
            param.trainable = False

def finetune_train(dump_dir, output_dir, max_step, batch_size=None, learning_rate=None):
    # 读取默认的yaml文件
    with open(config_path) as f:
        finetune_config = yaml.safe_load(f)
    # 多少个 step 保存一次
    # 1. 自动调整 batch ，通过 dump/train里面文件数量判断
    if not batch_size:
        train_data_dir = os.path.join(dump_dir, "train/norm/data_speech")
        file_num = len([filename for filename in os.listdir(train_data_dir) if filename.endswith(".npy")])
        if file_num <= 32:
            batch_size = file_num
            finetune_config['batch_size'] = batch_size
        else:
            finetune_config['batch_size'] = 32
    else:
        finetune_config['batch_size'] = batch_size

    # 2. 支持调整 learning_rate
    if learning_rate:
        finetune_config['learning_rate'] = learning_rate

    # 重新生成这次试验需要的yaml文件
    new_config_path = os.path.join(dump_dir, "finetune.yaml")
    with open(new_config_path, "w", encoding="utf8") as f:
        yaml.dump(finetune_config, f)


    train_metadata = f"{dump_dir}/train/norm/metadata.jsonl"
    dev_metadata = f"{dump_dir}/dev/norm/metadata.jsonl"
    speaker_dict = f"{dump_dir}/dump/speaker_id_map.txt"
    phones_dict = f"{dump_dir}/dump/phone_id_map.txt"
    num_workers = 2

    default_config_file = f"{pretrained_model_dir}/default.yaml"
    with open(default_config_file) as f:
        config = CfgNode(yaml.safe_load(f))

    # 冻结神经层
    with open(new_config_path) as f2:
            finetune_config = CfgNode(yaml.safe_load(f2))
    config.batch_size = finetune_config.batch_size if finetune_config.batch_size > 0 else config.batch_size
    config.optimizer.learning_rate = finetune_config.learning_rate if finetune_config.learning_rate > 0 else config.optimizer.learning_rate
    config.num_snapshots = finetune_config.num_snapshots if finetune_config.num_snapshots > 0 else config.num_snapshots
    frozen_layers = finetune_config.frozen_layers

    fields = [
            "text", "text_lengths", "speech", "speech_lengths", "durations",
            "pitch", "energy"
        ]
    converters = {"speech": np.load, "pitch": np.load, "energy": np.load}
    collate_fn = fastspeech2_multi_spk_batch_fn
    with open(speaker_dict, 'rt') as f:
        spk_id = [line.strip().split() for line in f.readlines()]
    spk_num = len(spk_id)
    fields += ["spk_id"]

    with jsonlines.open(train_metadata, 'r') as reader:
        train_metadata = list(reader)
    train_dataset = DataTable(
        data=train_metadata,
        fields=fields,
        converters=converters, )
    with jsonlines.open(dev_metadata, 'r') as reader:
        dev_metadata = list(reader)
    dev_dataset = DataTable(
        data=dev_metadata,
        fields=fields,
        converters=converters, )
    train_batch_size = min(len(train_metadata), batch_size)
    train_sampler = DistributedBatchSampler(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers)

    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,   # 输入 batch size 大小
        collate_fn=collate_fn,
        num_workers=num_workers)
    print("dataloaders done!")

    with open(phones_dict, "r") as f:
        phn_id = [line.strip().split() for line in f.readlines()]
    vocab_size = len(phn_id)
    print("vocab_size:", vocab_size)
    odim = config.n_mels
    
    # 初始化模型，优化器，损失函数
    model = FastSpeech2(
        idim=vocab_size, odim=odim, spk_num=spk_num, **config["model"])
    optimizer = build_optimizers(model, **config["optimizer"])
    use_masking=config["updater"]['use_masking']
    use_weighted_masking=False
    criterion = FastSpeech2Loss(use_masking=use_masking, use_weighted_masking=use_weighted_masking)
    
    # 检查之前是否有模型是否存在
    output_checkpoints_dir = os.path.join(output_dir, "checkpoints")
    if os.path.exists(output_checkpoints_dir):
        ckpt = find_max_ckpt(output_checkpoints_dir)
        if ckpt != 99200:
            use_pretrain_model = os.path.join(output_checkpoints_dir, "snapshot_iter_{ckpt}.pdz")
            start_step = ckpt
        else:
            cmd = f"rm -rf {output_checkpoints_dir}/*.pdz"
            os.system(cmd)
            use_pretrain_model = os.path.join(pretrained_model_dir, "snapshot_iter_99200.pdz")
            start_step = 0
    else:
        os.makedirs(output_checkpoints_dir, exist_ok=True)
        use_pretrain_model = os.path.join(pretrained_model_dir, "snapshot_iter_99200.pdz")
        start_step = 0
    
    # 加载预训练模型
    archive = paddle.load(use_pretrain_model)
    model.set_state_dict(archive['main_params'])
    optimizer.set_state_dict(archive['main_optimizer'])
    

    # 冻结层
    if frozen_layers != []:
        freeze_layer(model, frozen_layers)
    
    # 开始训练
    step = start_step

    # 进入训练流程
    while True:
        for batch_id, batch in enumerate(train_dataloader()):
            #前向计算的过程
            losses_dict = {}
            # spk_id!=None in multiple spk fastspeech2 
            spk_id = batch["spk_id"] if "spk_id" in batch else None
            spk_emb = batch["spk_emb"] if "spk_emb" in batch else None
            # No explicit speaker identifier labels are used during voice cloning training.
            if spk_emb is not None:
                spk_id = None

            before_outs, after_outs, d_outs, p_outs, e_outs, ys, olens = model(
                text=batch["text"],
                text_lengths=batch["text_lengths"],
                speech=batch["speech"],
                speech_lengths=batch["speech_lengths"],
                durations=batch["durations"],
                pitch=batch["pitch"],
                energy=batch["energy"],
                spk_id=spk_id,
                spk_emb=spk_emb)
            
            l1_loss, duration_loss, pitch_loss, energy_loss = criterion(
                after_outs=after_outs,
                before_outs=before_outs,
                d_outs=d_outs,
                p_outs=p_outs,
                e_outs=e_outs,
                ys=ys,
                ds=batch["durations"],
                ps=batch["pitch"],
                es=batch["energy"],
                ilens=batch["text_lengths"],
                olens=olens)

            loss = l1_loss + duration_loss + pitch_loss + energy_loss

            # optimizer = self.optimizer
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            
            losses_dict["l1_loss"] = float(l1_loss)
            losses_dict["duration_loss"] = float(duration_loss)
            losses_dict["pitch_loss"] = float(pitch_loss)
            losses_dict["energy_loss"] = float(energy_loss)
            losses_dict["loss"] = float(loss)
            msg = f"Step: {step}, " + ', '.join('{}: {:>.6f}'.format(k, v)
                                for k, v in losses_dict.items())
            
            print(msg)
            yield msg
            if step > max_step:
                # 保存模型
                save_path = f"{output_checkpoints_dir}/snapshot_iter_{step}.pdz"
                archive = {
                    "epoch": 100,
                    "iteration": step,
                    "main_params": model.state_dict(),
                    "main_optimizer": optimizer.state_dict()
                }
                paddle.save(archive, save_path)

# 全流程微调训练
def finetuneTTS(label_path, exp_name, start_step=1, end_step=100, epoch=100, batch_size=None, learning_rate=None):
    exp_path = os.path.join(exp_base + exp_name)
    os.makedirs(exp_path, exist_ok=True)
    data_dir = os.path.join(exp_path, "data")
    finetune_status_json = os.path.join(exp_path, "finetune_status.json")
    status_change(finetune_status_json, True)
    
    if start_step <= 1 and end_step > 1:
        # Step1 : 生成标准数据集
        if not os.path.exists(data_dir):
            if not step1_generate_standard_dataset(label_path, exp_path, data_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{data_dir} 已存在，跳过此步骤！")
    
    # Step2: 检查非法数据
    new_dir = os.path.join(exp_path, "new_dir")
    if start_step <= 2 and end_step >= 2:  
        if not os.path.exists(new_dir):
            if not step2_check_oov(data_dir, new_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{new_dir} 已存在，跳过此步骤")
    
    mfa_dir = os.path.join(exp_path, "mfa")
    if start_step <= 3 and end_step >= 3:
        # Step3: MFA 对齐
        if not os.path.exists(mfa_dir):
            if not step3_get_mfa(new_dir, mfa_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{mfa_dir} 已存在，跳过此步骤")
    
    if start_step <= 4 and end_step >= 4:
        # Step4: 生成时长信息文件
        if not step4_duration(mfa_dir):
            status_change(finetune_status_json, False)
            return
    
    dump_dir = os.path.join(exp_path, "dump")
    if start_step <= 5 and end_step >= 5:
        # Step5: 数据预处理
        if not os.path.exists(dump_dir):
            if not step5_extract_feature(data_dir, dump_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{dump_dir} 已存在，跳过此步骤")
    
    output_dir = os.path.join(exp_path, "output")
    if start_step <= 6 and end_step >= 6:
        # Step6: 生成训练环境
        if not os.path.exists(output_dir):
            if not step6_prepare_env(output_dir):
                status_change(finetune_status_json, False)
                return
        else:
            print(f"{output_dir} 已存在，跳过此步骤")
    
    if start_step <= 7 and end_step >= 7:
        # Step7: 微调训练
        if not step7_finetune(dump_dir, output_dir, epoch, batch_size=None, learning_rate=None):
            status_change(finetune_status_json, False)
            return
    
    if start_step <= 8 and end_step >= 8:
        # Step8: 导出静态图模型
        if not step8_export_static_model(exp_name):
            status_change(finetune_status_json, False)
            return
    status_change(finetune_status_json, False)
    return True

# 生成音频
def generateTTS(text_dict, exp_name, voc="PWGan"):
    if not step9_generateTTS_inference(text_dict, exp_name, voc):
        print("音频生成失败，请微调模型是否成功!")
        return None
    else:
        return True
