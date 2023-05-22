# SNet-X
Using Convolution-X (LMMSE-Conv & Diffusion-Conv) and joint-training allows the model to remove speckle noise of SAR image more efficiently.

SNet-X is applicable to any polarization mode and can denoise each polarimetric channel of the SAR image one by one.

The paper will be published in IEEE IGARSS-2023, and the link will be updated subsequently. For more details, please refer to the paper.

This code base provides the following test images：

|      SAR      | Frequency / GHz |     Mode    | Resolution / m (Slant range) |  Platform  | Height / km |
| ------------- | --------------- | ----------- | ---------------------------- | ---------- | ----------- |
|     CP-SAR    |   5.3 (C-band)  | Circular RR |            0.375             |  Airborne  |      1      |
|     ERS-1     |   5.3 (C-band)  |  Linear VV  |            9.677             | Spaceborne |     775     |
|  Sentinel-1A  |   5.4 (C-band)  |  Linear VH  |            20.50             | Spaceborne |     693     |
| Virtual-SAR-A |     Optical     |     RGB     |            30-0.2            |  Airborne  |      3*     |
| Virtual-SAR-B |     Optical     |     RGB     |            30-0.2            |  Airborne  |      3*     |

The resolution and shooting height are not fixed in the Virtual-SAR dataset, so the data of it shown in the table are for reference only.

The systematic parameters of SAR (resolution and shooting height / scale of the scene) affect the denoising effect of the model.

The SNet-X-A model is trained using the Virtual-SAR dataset, and the results are applicable to the denoising of Sentinel-1A data.

The SNet-X-B model is trained with the adjusted Virtul-SAR dataset (Adjustment.py), and the results are applicable to the denoising of CP-SAR, ERS-1 and GAOFEN-3 data.

Evaluation is divided into Virtual Evaluation (Judgement-Virtual.py) and Real Evaluation (M-index & TeacherNet).

The Level-set method can be used to compare with the denoising evaluation of TeacherNet.

This library also provides the following speckle noise filtering methods：

Local filter：

Median (?), Lee (Lee et al. 1981), Frost (Frost et al. 1982), Kuan (Kuan et al. 1985), GAMMA-MAP (Lopes et al. 1990).

Non-Local filter：

NLM (Buades et al. 2005), PPB (Deledalle et al. 2009), SAR-BM3D (Parrilli et al. 2012), FANS (Cozzolino et al. 2014), MuLog-BM3D (Deledalle et al. 2017), RABASAR (Zhao et al. 2019).

Wavelet filter：

POAC (Mastriani et al. 2016).

Supervised learning:

SAR-CNN (Chierchia et al. 2017), ID-CNN (Wang et al. 2017), SAR-DRN (Zhang et al. 2018), DoPAMINE (Joo et al. 2019), MONet (Vitale et al. 2021), SD-Net (Cai et al. 2021), deSpeckNet (Mullissa et al. 2022), SNet-X (Cai et al. 2023).

Semi-supervised learning:

Noise2Noise (Lehtinen et al. 2018), SAR2SAR (Dalsasso et al. 2021), Speckle2Void (Molini et al. 2022).

Please refer to the relevant literature for the details of the algorithm.
