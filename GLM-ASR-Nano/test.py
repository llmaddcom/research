"""
GLM-ASR-Nano 音频转录测试脚本

该脚本实现了基于 GLM-ASR-Nano 模型的音频转文字功能，支持：
- 长音频文件的自动切割处理
- 音频特征提取和token化
- 批量音频片段转录
- 临时文件管理
"""

import argparse  # 命令行参数解析（当前未使用，但保留用于未来扩展）
import shutil  # 文件和目录操作，用于清理临时文件
import tempfile  # 临时文件和目录创建
from pathlib import Path  # 路径操作

import torch  # PyTorch深度学习框架
import torchaudio  # PyTorch音频处理库
from transformers import (
    AutoConfig,  # 自动加载模型配置
    AutoModelForCausalLM,  # 自动加载因果语言模型（用于ASR）
    AutoTokenizer,  # 自动加载分词器
    WhisperFeatureExtractor,  # Whisper特征提取器（用于音频特征提取）
)

# Whisper特征提取器的配置参数
# 这些参数定义了音频预处理的方式，与Whisper模型的输入要求一致
WHISPER_FEAT_CFG = {
    "chunk_length": 30,  # 每个音频块的长度（秒）
    "feature_extractor_type": "WhisperFeatureExtractor",  # 特征提取器类型
    "feature_size": 128,  # 特征维度大小（Mel频谱图的频率bins数）
    "hop_length": 160,  # 帧移长度（样本数），决定时间分辨率
    "n_fft": 400,  # FFT窗口大小（样本数），用于计算频谱
    "n_samples": 480000,  # 每个chunk的样本数（30秒 * 16000采样率）
    "nb_max_frames": 3000,  # 最大帧数限制
    "padding_side": "right",  # 填充方向（右侧填充）
    "padding_value": 0.0,  # 填充值
    "processor_class": "WhisperProcessor",  # 处理器类别
    "return_attention_mask": False,  # 不返回注意力掩码
    "sampling_rate": 16000,  # 采样率（Hz），Whisper标准采样率
}


def get_audio_token_length(seconds, merge_factor=2):
    """
    根据音频时长计算对应的音频token数量
    
    该函数模拟了音频经过CNN特征提取和token合并后的长度计算过程。
    计算流程：
    1. 将音频时长转换为Mel频谱图长度（100帧/秒）
    2. 模拟CNN卷积层对序列长度的压缩
    3. 根据merge_factor计算最终token数量
    4. 应用最大长度限制
    
    Args:
        seconds: 音频时长（秒）
        merge_factor: token合并因子，默认为2（每2个特征合并为1个token）
    
    Returns:
        int: 音频对应的token数量
    """
    def get_T_after_cnn(L_in, dilation=1):
        """
        计算音频特征经过CNN卷积层后的序列长度
        
        模拟两层CNN卷积：
        - 第一层：padding=1, kernel_size=3, stride=1（不改变长度）
        - 第二层：padding=1, kernel_size=3, stride=2（长度减半）
        
        Args:
            L_in: 输入序列长度
            dilation: 卷积膨胀率，默认为1
        
        Returns:
            int: 经过CNN后的序列长度
        """
        # CNN层配置：[(padding, kernel_size, stride), ...]
        # 第一层：padding=1, kernel=3, stride=1（保持长度）
        # 第二层：padding=1, kernel=3, stride=2（长度减半）
        for padding, kernel_size, stride in eval("[(1,3,1)] + [(1,3,2)] "):
            # 卷积输出长度计算公式
            L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            L_out = 1 + L_out // stride
            L_in = L_out
        return L_out

    # 将音频时长转换为Mel频谱图长度（100帧/秒，即每10ms一帧）
    mel_len = int(seconds * 100)
    
    # 计算经过CNN后的序列长度
    audio_len_after_cnn = get_T_after_cnn(mel_len)
    
    # 根据merge_factor计算token数量
    # merge_factor表示每N个特征合并为1个token
    audio_token_num = (audio_len_after_cnn - merge_factor) // merge_factor + 1

    # TODO: 当前whisper模型无法处理更长的序列，未来可能需要切割chunk
    # 限制最大token数量，避免超出模型处理能力
    audio_token_num = min(audio_token_num, 1500 // merge_factor)

    return audio_token_num


def split_audio_file(
    audio_path: str | Path,
    chunk_seconds: int = 30,
    output_format: str = "wav",
    sampling_rate: int = 16000,
) -> tuple[list[Path], Path]:
    """
    将音频文件按指定秒数切割成多个临时文件
    
    该函数用于处理长音频文件，将其分割成多个固定时长的片段，便于模型逐段处理。
    支持多种音频格式（MP3、WAV等），自动处理多声道和采样率转换。
    
    Args:
        audio_path: 音频文件路径（支持MP3、WAV等torchaudio支持的格式）
        chunk_seconds: 每个切割片段的秒数，默认30秒
        output_format: 输出文件格式，默认"wav"
        sampling_rate: 输出采样率，默认16000Hz（Whisper标准采样率）
    
    Returns:
        tuple: (临时文件路径列表, 临时目录路径)
            - 临时文件路径列表：按时间顺序排列的切割后音频文件路径
            - 临时目录路径：存放所有临时文件的目录路径
    
    Raises:
        FileNotFoundError: 当音频文件不存在时
        RuntimeError: 当切割过程中出现错误时
    """
    # 转换为Path对象便于操作
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    
    # 使用torchaudio加载音频文件
    # wav: 音频数据张量，形状为 [channels, samples]
    # sr: 原始采样率
    wav, sr = torchaudio.load(str(audio_path))
    
    # 如果是多声道音频，只保留第一个声道（转为单声道）
    # 模型通常只需要单声道输入
    if wav.shape[0] > 1:
        wav = wav[:1, :]
    
    # 如果原始采样率与目标采样率不同，进行重采样
    # 确保所有音频片段使用统一的采样率
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        wav = resampler(wav)
        sr = sampling_rate
    
    # 计算每个片段的样本数（采样率 * 秒数）
    chunk_samples = chunk_seconds * sampling_rate
    total_samples = wav.shape[1]  # 获取音频总样本数
    
    # 创建临时文件列表和临时目录
    temp_files = []
    # 创建临时目录，前缀为"audio_chunks_"，便于识别
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
    
    try:
        # 按固定步长切割音频
        chunk_idx = 0
        # 从0开始，每次前进chunk_samples个样本
        for start in range(0, total_samples, chunk_samples):
            # 计算当前片段的结束位置（不超过总长度）
            end = min(start + chunk_samples, total_samples)
            # 提取音频片段 [channel, start:end]
            chunk = wav[:, start:end]
            
            # 创建临时文件路径，使用4位数字编号（0000, 0001, ...）
            temp_file = temp_dir / f"chunk_{chunk_idx:04d}.{output_format}"
            temp_files.append(temp_file)
            
            # 保存音频片段到临时文件
            torchaudio.save(
                str(temp_file),  # 文件路径
                chunk,  # 音频数据
                sampling_rate,  # 采样率
                format=output_format,  # 文件格式
            )
            
            chunk_idx += 1
        
        print(f"音频已切割为 {len(temp_files)} 个片段，保存在: {temp_dir}")
        return temp_files, temp_dir
    
    except Exception as e:
        # 如果切割过程中出现任何错误，清理已创建的临时文件
        cleanup_temp_files(temp_dir)
        raise RuntimeError(f"切割音频文件时出错: {e}")


def cleanup_temp_files(temp_dir: str | Path):
    """
    清理临时文件和目录
    
    该函数用于删除之前创建的临时音频文件及其目录，释放磁盘空间。
    通常在音频处理完成后调用。
    
    Args:
        temp_dir: 临时目录路径（可以是字符串或Path对象）
    
    Note:
        如果目录不存在或删除失败，函数会打印错误信息但不会抛出异常，
        确保主流程不会因清理失败而中断。
    """
    temp_dir = Path(temp_dir)
    # 检查目录是否存在且确实是目录
    if temp_dir.exists() and temp_dir.is_dir():
        try:
            # 递归删除整个目录及其所有内容
            shutil.rmtree(temp_dir)
            print(f"已清理临时目录: {temp_dir}")
        except Exception as e:
            # 如果删除失败（如文件被占用），打印错误但不抛出异常
            print(f"清理临时目录时出错: {e}")


def build_prompt(
    audio_path: Path,
    tokenizer,
    feature_extractor: WhisperFeatureExtractor,
    merge_factor: int,
    chunk_seconds: int = 30,
) -> dict:
    """
    构建模型输入的prompt和音频特征
    
    该函数将音频文件转换为模型所需的输入格式，包括：
    1. 加载和预处理音频（单声道、重采样）
    2. 将音频切分为chunk并提取Mel频谱特征
    3. 构建包含特殊token的文本prompt
    4. 记录音频token的位置和长度信息
    
    Args:
        audio_path: 音频文件路径
        tokenizer: 分词器对象，用于编码文本token
        feature_extractor: Whisper特征提取器，用于提取音频Mel频谱特征
        merge_factor: token合并因子，用于计算音频token数量
        chunk_seconds: 每个音频chunk的秒数，默认30秒
    
    Returns:
        dict: 包含以下键的批次数据字典
            - input_ids: 文本token序列（包含特殊token和占位符）
            - audios: 音频Mel频谱特征张量
            - audio_offsets: 每个音频chunk在token序列中的起始位置
            - audio_length: 每个音频chunk对应的token数量
            - attention_mask: 注意力掩码（全1，表示所有位置都参与计算）
    
    Raises:
        ValueError: 当音频内容为空或加载失败时
    """
    # 确保audio_path是Path对象
    audio_path = Path(audio_path)
    
    # 加载音频文件
    wav, sr = torchaudio.load(str(audio_path))
    # 只保留第一个声道（转为单声道）
    wav = wav[:1, :]
    
    # 如果采样率不匹配，进行重采样
    if sr != feature_extractor.sampling_rate:
        wav = torchaudio.transforms.Resample(sr, feature_extractor.sampling_rate)(wav)

    # 初始化token列表，用于构建完整的prompt序列
    tokens = []
    # 添加用户角色标记
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\n")

    # 初始化音频相关列表
    audios = []  # 存储每个chunk的Mel频谱特征
    audio_offsets = []  # 存储每个音频chunk在token序列中的起始位置
    audio_length = []  # 存储每个音频chunk对应的token数量
    
    # 计算每个chunk的样本数
    chunk_size = chunk_seconds * feature_extractor.sampling_rate
    
    # 将音频切分为多个chunk并处理
    for start in range(0, wav.shape[1], chunk_size):
        # 提取当前chunk的音频数据
        chunk = wav[:, start : start + chunk_size]
        
        # 使用Whisper特征提取器提取Mel频谱特征
        # 输入需要是numpy数组格式
        mel = feature_extractor(
            chunk.numpy(),  # 转换为numpy数组
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",  # 返回PyTorch张量
            padding="max_length",  # 填充到最大长度
        )["input_features"]  # 提取input_features字段
        audios.append(mel)
        
        # 计算当前chunk的时长（秒）
        seconds = chunk.shape[1] / feature_extractor.sampling_rate
        # 计算该chunk对应的音频token数量
        num_tokens = get_audio_token_length(seconds, merge_factor)
        
        # 在token序列中添加音频开始标记
        tokens += tokenizer.encode("<|begin_of_audio|>")
        # 记录音频token的起始位置（在添加占位符之前）
        audio_offsets.append(len(tokens))
        # 添加占位符token（用0表示，实际音频特征在audios中）
        tokens += [0] * num_tokens
        # 添加音频结束标记
        tokens += tokenizer.encode("<|end_of_audio|>")
        # 记录该chunk的token数量
        audio_length.append(num_tokens)

    # 检查是否成功提取了音频特征
    if not audios:
        raise ValueError("音频内容为空或加载失败。")

    # 添加用户指令部分
    tokens += tokenizer.encode("<|user|>")
    tokens += tokenizer.encode("\nPlease transcribe this audio into text")

    # 添加助手回复标记（模型将从此处开始生成转录文本）
    tokens += tokenizer.encode("<|assistant|>")
    tokens += tokenizer.encode("\n")

    # 构建批次数据字典
    batch = {
        "input_ids": torch.tensor([tokens], dtype=torch.long),  # 文本token序列，添加batch维度
        "audios": torch.cat(audios, dim=0),  # 将所有chunk的音频特征拼接
        "audio_offsets": [audio_offsets],  # 音频位置信息，添加batch维度
        "audio_length": [audio_length],  # 音频长度信息，添加batch维度
        "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),  # 注意力掩码（全1）
    }
    return batch


def prepare_inputs(batch: dict, device: torch.device) -> tuple[dict, int]:
    """
    准备模型输入数据，将数据移动到指定设备并转换数据类型
    
    该函数将批次数据从CPU移动到GPU（或其他指定设备），
    并对音频特征进行数据类型转换以节省显存。
    
    Args:
        batch: 包含以下键的批次数据字典
            - input_ids: 文本token序列
            - attention_mask: 注意力掩码
            - audios: 音频Mel频谱特征
            - audio_offsets: 音频位置信息
            - audio_length: 音频长度信息
        device: 目标设备（如 "cuda:0" 或 torch.device("cuda:0")）
    
    Returns:
        tuple: (模型输入字典, prompt长度)
            - 模型输入字典：包含移动到设备并转换类型后的所有输入
            - prompt长度：输入序列的长度（用于后续截取生成的文本）
    """
    # 将文本token序列移动到指定设备
    tokens = batch["input_ids"].to(device)
    # 将注意力掩码移动到指定设备
    attention_mask = batch["attention_mask"].to(device)
    # 将音频特征移动到指定设备
    audios = batch["audios"].to(device)
    
    # 构建模型输入字典
    model_inputs = {
        "inputs": tokens,  # 文本token序列
        "attention_mask": attention_mask,  # 注意力掩码
        "audios": audios.to(torch.bfloat16),  # 音频特征转换为bfloat16以节省显存
        "audio_offsets": batch["audio_offsets"],  # 音频位置信息（列表，不需要移动）
        "audio_length": batch["audio_length"],  # 音频长度信息（列表，不需要移动）
    }
    # 返回模型输入和prompt长度（tokens的序列长度）
    return model_inputs, tokens.size(1)


def transcribe(
    checkpoint_dir: Path,
    audio_path: Path,
    tokenizer_path: str,
    max_new_tokens: int,
    device: str,
):
    """
    转录音频文件为文本
    
    这是主要的转录函数，完成以下流程：
    1. 加载模型、分词器和特征提取器
    2. 构建包含音频特征的prompt
    3. 使用模型生成转录文本
    4. 解码并保存转录结果
    
    Args:
        checkpoint_dir: 模型检查点目录路径（包含模型权重和配置）
        audio_path: 要转录的音频文件路径
        tokenizer_path: 分词器路径（如果为None则使用checkpoint_dir）
        max_new_tokens: 最大生成token数量（限制转录文本长度）
        device: 计算设备（如 "cuda:0" 或 "cpu"）
    
    Note:
        - 转录结果会追加写入到 "transcript.txt" 文件中
        - 如果转录为空，会打印 "[Empty transcription]"
    """
    # 确定分词器的加载路径（如果未指定则使用模型目录）
    tokenizer_source = tokenizer_path if tokenizer_path else checkpoint_dir
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    # 创建Whisper特征提取器，使用预定义的配置
    feature_extractor = WhisperFeatureExtractor(**WHISPER_FEAT_CFG)

    # 加载模型配置（需要trust_remote_code=True因为使用了自定义代码）
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    # 加载模型
    # 使用bfloat16精度以节省显存并加速推理
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        config=config,
        torch_dtype=torch.bfloat16,  # 使用bfloat16半精度
        trust_remote_code=True,  # 信任远程代码（自定义模型架构）
    ).to(device)  # 将模型移动到指定设备
    # 设置为评估模式（禁用dropout等训练特性）
    model.eval()

    # 构建包含音频特征的prompt批次数据
    batch = build_prompt(
        audio_path,
        tokenizer,
        feature_extractor,
        merge_factor=config.merge_factor,  # 从配置中获取merge_factor
    )

    # 准备模型输入（移动到设备、转换数据类型）
    model_inputs, prompt_len = prepare_inputs(batch, device)

    # 使用推理模式（禁用梯度计算，节省显存和加速）
    with torch.inference_mode():
        # 使用模型生成转录文本
        generated = model.generate(
            **model_inputs,  # 展开输入字典
            max_new_tokens=max_new_tokens,  # 最大生成token数
            do_sample=False,  # 使用贪婪解码（不采样，选择概率最高的token）
        )
    
    # 提取生成的token（去掉prompt部分，只保留新生成的内容）
    # generated[0, prompt_len:] 表示第一个batch，从prompt_len开始到结尾
    transcript_ids = generated[0, prompt_len:].cpu().tolist()
    # 将token ID解码为文本，跳过特殊token
    transcript = tokenizer.decode(transcript_ids, skip_special_tokens=True).strip()
    
    # 将转录结果追加写入文件（使用追加模式"a"）
    with open("transcript1.txt", "a", encoding="utf-8") as f:
        f.write(transcript)
    
    # 打印分隔线和转录结果
    print("----------")
    print(transcript or "[Empty transcription]")




if __name__ == "__main__":
    """
    主程序入口
    
    该脚本的主要工作流程：
    1. 配置模型路径、音频路径和设备
    2. 将长音频文件切割成多个短片段（避免超出模型处理能力）
    3. 对每个音频片段进行转录
    4. 清理临时文件
    
    使用场景：
    - 处理长音频文件（超过模型单次处理能力）
    - 批量处理多个音频片段
    - 将转录结果追加保存到 transcript.txt 文件
    """
    # ========== 配置参数 ==========
    checkpoint_dir = "/data/disk1/model/GLM-ASR-Nano-2512"  # 模型检查点目录
    audio_path = "/data/zh/内容制造平台-24年8月8日.mp3"  # 要转录的音频文件路径
    device = "cuda:6"  # 使用的GPU设备编号
    
    # 如果音频文件较短，可以直接转录（不需要切割）
    # transcribe(checkpoint_dir, audio_path, tokenizer_path=checkpoint_dir, max_new_tokens=128, device="cuda:6")
    
    # ========== 音频切割 ==========
    # 将长音频文件切割成多个3秒的片段
    # 切割后的文件保存在临时目录中
    temp_files, temp_dir = split_audio_file(
        audio_path=audio_path,
        chunk_seconds=10,  # 每个片段3秒（可根据需要调整）
    )
    
    # ========== 批量转录 ==========
    # 对每个切割后的音频片段进行转录
    # 转录结果会追加写入到 transcript.txt 文件中
    for temp_file in temp_files:
        transcribe(
            checkpoint_dir, 
            temp_file, 
            tokenizer_path=checkpoint_dir, 
            max_new_tokens=128,  # 最大生成token数（可根据需要调整）
            device=device
        )
    
    # ========== 清理临时文件 ==========
    # 转录完成后，删除临时目录和所有临时文件，释放磁盘空间
    cleanup_temp_files(temp_dir)


    # 依托答辩