# <p align=center>`Awesome Remote Sensing Large Model`</p>

Awesome Remote Sensing Large Model：
Visual Remote Sensing Foundation Models and Visual-Language Remote Sensing Goundation Models

# Visual-Language Remote Sensing Goundation Models 视觉-语言遥感基础模型

- 2024-arxiv. [**Paper**](https://arxiv.org/abs/2410.07167) | [**github(code)**](https://github.com/shikiw/Modality-Integration-Rate) 

    Deciphering Cross-Modal Alignment in Large Vision-Language Models with Modality Integration Rate
    </br>本文提出了模态整合率MIR，从分布距离的角度衡量LVLMs的预训练质量，具有高效性，对数据样本具有鲁棒性，
    对模型结构及训练方法泛化性。同时提出了一个轻量级的、可学习的视觉tokens校准模块MoCa，旨在增强视觉tokens与文本tokens的对齐。

- 2024-TGRS. [**EarthGPT(Paper )**](https://ieeexplore.ieee.org/document/10547418) | [**github(no code)**](https://github.com/wivizhang/EarthGPT) 

    EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain
    </br>EarthGPT 本文提出了一种名为 EarthGPT 的先驱 MLLM，它将各种多传感器遥感解释任务统一集成在一起，以实现通用的遥感图像理解。
    首先，**构建了一种视觉增强感知机制**，以细化和整合粗尺度语义感知信息和细尺度细节感知信息。
    其次，**提出了一种跨模态相互理解方法**，旨在增强视觉感知和语言理解之间的相互作用，加深对视觉和语言内容的理解。
    最后，**提出了一种用于遥感领域多传感器多任务的统一指令调整方法**，以统一包括场景分类、图像字幕、区域级字幕、视觉问答 (VQA)、视觉基础和物体检测在内的广泛任务。
    更重要的是，构建了**大规模多传感器多模态遥感指令跟踪数据集MMRS-1M**，该数据集基于现有的34个多样化遥感数据集，包含超过100万个图像-文本对，包括光学、合成孔径雷达（SAR）和红外等多传感器图像。
    MMRS-1M数据集解决了MLLM在遥感专家知识方面的缺陷，并促进了MLLM在遥感领域的发展。 
    我们进行了大量的实验，证明了 EarthGPT 在各种 RS 视觉解释任务中的表现优于其他专业模型和 MLLM，证明了所提出的 EarthGPT 的有效性，并为开放集推理任务提供了通用范例。

- 2024-TGRS.[**RS5M(Paper)**](https://ieeexplore.ieee.org/document/10679571) | [**code**](https://github.com/om-ai-lab/RS5M) 
    
    RS5M: A Large Scale Vision-Language Dataset for Remote Sensing Vision-Language Model
    </br>在本文中，我们提出了一个包含领域预训练视觉语言模型 (DVLM) 的新框架，弥合了通用视觉语言模型 (GVLM) 与领域特定下游任务之间的差距。
    此外，我们提出了一个**遥感 (RS) 领域的图文配对数据集 RS5M**，其中包含 **500 万张带有英文描述的遥感图像**。
    该数据集是通过使用预训练的 VLM 筛选公开可用的图像文本配对数据集和仅带标签的 RS 数据集而获得的。
    这些数据集构成了第一个大规模 RS 图像文本配对数据集。此外，我们对 CLIP 模型进行了微调，并在 RS5M 上尝试了几种参数高效的微调方法来实现 DVLM。
    实验结果表明，我们提出的数据集对各种任务都非常有效，并且我们的模型 GeoRSCLIP 在零样本分类 (ZSC) 中比基线或之前最先进的模型提高了 3%∼20% ，
    在遥感跨模态文本图像检索 (RSCTIR) 中提高了3%∼6% ，在语义定位 (SeLo) 任务中提高了4%∼5%。

- 2024-CVPR.[**GeoChat(Paper)**](https://arxiv.org/abs/2311.15826) | [**code**](https://github.com/mbzuai-oryx/geochat) 
      
    GeoChat:Grounded Large Vision-Language Model for Remote Sensing
    </br>我们提出了 **GeoChat - 第一个多功能遥感 VLM**，它提供**高分辨率 RS 图像的多任务对话功能**。
    具体来说，GeoChat 不仅可以回答图像级查询，还可以接受区域输入以进行区域特定对话。此外，它可以通过参考对象的空间坐标在其响应中直观地定位对象。
    为了解决缺乏特定领域数据集的问题，我们通过扩展现有多样化 RS 数据集中的图像-文本对，生成了一个新的 **RS 多模态指令跟踪数据集**。
    我们为 RS 多任务对话建立了一个全面的基准，并与许多基线方法进行了比较。GeoChat 在各种 RS 任务上展示了强大的零样本性能，
    例如图像和区域字幕、视觉问答、场景分类、基于视觉的对话和指称检测。

- 2023.[**GRAFT(Paper)**](https://arxiv.org/abs/2312.06960) 
      
    Remote Sensing Vision-Language Foundation Models without Annotations via Ground Remote Alignment
    </br>使用大量配对的互联网和卫星图像训练遥感图像的图像编码器，使其与 CLIP 的图像编码器对齐。
    无监督方法能够为两种不同分辨率的遥感图像训练出首创的大规模视觉语言模型 (VLM)。
    实验表明，这些 VLM 可实现卫星图像的零样本、开放词汇图像分类、检索、分割和视觉问答。在这些任务中的每一个任务中，
    在没有文本注释的情况下训练的 VLM 都优于现有的经过监督训练的 VLM，分类性能提高 20%，分割性能提高 80%。

- 2023-arxiv.[**RSGPT(Paper)**](https://arxiv.org/abs/2307.15266) | [**github(no code)**](https://github.com/Lavender105/RSGPT) 
      
    RSGPT: A Remote Sensing Vision Language Model and Benchmark
    </br>在这项工作中，我们**构建了一个高质量的遥感图像字幕数据集 (RSICap)**，以促进 RS 领域大型 VLM 的开发。
    与之前使用模型生成的字幕或简短描述的 RS 数据集不同，RSICap 包含 **2,585** 个人工注释的字幕，具有丰富且高质量的信息。
    该数据集为每幅图像提供详细描述，包括场景描述（例如住宅区、机场或农田）以及对象信息（例如颜色、形状、数量、绝对位置等）。
    为了促进 RS 领域 VLM 的评估，我们还提供了一个名为 RSIEval 的基准评估数据集。 
    该数据集由人工注释的字幕和视觉问答对组成，可在 RS 背景下对 VLM 进行全面评估。

# Visual Remote Sensing Foundation Models  视觉遥感基础模型

- 2024-IEEE JSTARS Special issue.[**MTP(Paper)**](https://arxiv.org/abs/2403.13430) | [**github(code)**](https://github.com/ViTAE-Transformer/MTP)     
    
    MTP: Advancing Remote Sensing Foundation Model via Multi-Task Pretraining
    </br> 在本研究中，我们探索了 RS 基础模型的**多任务预训练 (MTP)** 范式来解决这个问题。使用共享编码器和特定于任务的解码器架构，
    我们对 SAMRS 数据集进行多任务监督预训练，包括语义分割、实例分割和旋转对象检测。MTP 支持卷积神经网络和视觉变换器基础模型，
    具有超过 3 亿个参数。预训练模型针对各种 RS 下游任务进行了微调，例如场景分类、水平和旋转对象检测、语义分割和变化检测。
    在 14 个数据集上进行的大量实验证明了我们的模型比现有的类似规模的模型更具优势，并且与更大的最先进模型相比具有竞争性能，
    从而验证了 MTP 的有效性。

- 2024-TGRS.[**RemoteCLIP(Paper)**](https://arxiv.org/abs/2306.11029) | [**github(no code)**](https://github.com/ChenDelong1999/RemoteCLIP) 
    
    RemoteCLIP: A Vision Language Foundation Model for Remote Sensing
    </br>提出了**RemoteCLIP**，这是第一个**用于遥感的视觉语言基础模型**，旨在学习具有丰富语义的稳健视觉特征以及对齐的文本嵌入，
    以实现无缝的下游应用。为了解决预训练数据的稀缺性，我们利用数据扩展，基于 Box-to-Caption (B2C) 和 Mask-to-Box (M2B) 转换转换异构注释，
    并进一步整合无人机图像，从而生成 12 倍大的预训练数据集。

- 2023.[**CrossEarth(Paper)**](https://arxiv.org/abs/2410.22629v2) | [**github(code)**](https://github.com/Cuzyoung/CrossEarth/)     
    
    CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable Remote Sensing Semantic Segmentation
    </br> 引入了第一个用于 RSDG **语义分割的视觉基础模型 CrossEarth**。
    CrossEarth 通过专门设计的数据级 Earth-Style 注入管道和模型级多任务训练管道展示了强大的跨域泛化能力。
    此外，对于语义分割任务，我们制定了一个 RSDG 基准，其中包含 28 个跨域设置，
    涵盖不同的区域、光谱带、平台和气候，为测试未来 RSDG 模型的通用性提供了全面的框架。
    对该基准进行的大量实验证明了 CrossEarth 优于现有的最先进方法。

- 2023.[**SkySense(Paper)**](https://arxiv.org/pdf/2312.10115)      

    SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery
    </br>提出了**SkySense，一个通用的十亿级模型**，经过预训练，使用经过策划的多模式遥感图像（RSI）数据集，包含2150万个时间序列。
    SkySense是**迄今为止最大的多模态遥感基础模型**，其模块可以灵活组合或单独使用以适应各种任务。它在广泛的评估中展现了显著的泛化能力，
    **涵盖了16个数据集，包括7个任务，从单模到多模、静态到时态、分类到定位**。SkySense在所有测试场景中均超过了18个最近的RSFM。
    具体而言，它在平均水平上分别超过了最新的模型，如GFM、SatLas和Scale-MAE，分别为2.76%、3.67%和3.61%。我们将发布预训练权重，以促进未来的研究和地球观测应用。

- 2022-TGRS.[**RingMo(Paper)**](https://ieeexplore.ieee.org/abstract/document/9844015)      
    
    RingMo: A Remote Sensing Foundation Model with Masked Image Modeling论文
    </br>在本文中，我们利用RS图像生成式自监督学习的优势，提出了一个名为**RingMo**的遥感大模型框架，该框架由两部分组成。
    首先，从卫星和航空平台收集**200万张覆盖全球多个场景和目标的遥感图像，构建大规模数据集**。
    其次，提出了一种针对复杂遥感场景中密集小目标的遥感基础模型训练方法。
    我们表明，使用RingMo方法在我们的数据集上训练的基础模型在跨四个下游任务的八个数据集上达到了最先进的水平，证明了所提出框架的有效性。
    通过深入探索，我们认为是时候让RS研究人员接受生成式自监督学习，并利用其通用表示能力来加速RS应用的开发。
