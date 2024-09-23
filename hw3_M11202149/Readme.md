## BLIP-2(CVPDL3)
## 建環境
```
Ubuntu 20.04
$ python=3.8
Pytorch版本:pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
pip install 一些需要的包
寫一個BLIP-2 Inference的code, BLIP-2.py（裡面可以直接選取7＊20張較適合GLIGEN inference的照片）
用BLIP-2的時候, OPT-2.7b產生的文字效果會比blip2-opt-6.7b-coco差 (輸出分別為output2.7b.json, output6.7b-coco.json)，所以選擇後者作為 better-generating results使用
用copy.py將140張原始圖片copy到./COCODIR/origintrain2017new資料夾
(template1(prompt1)為只有一行文字的)
(template2(prompt2)為多後面一些文字的)
## GLIGEN(CVPDL3)
## 建環境
```
$ pip install albumentations==0.4.3 opencv-python pudb==2019.2 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.4.2 omegaconf==2.1.1 test-tube>=0.7.5 streamlit>=0.73.1 einops==0.3.0 torch-fidelity==0.3.0 git+https://github.com/openai/CLIP.git protobuf~=3.20.1 torchmetrics==0.6.0 transformers==4.19.2 kornia==0.5.8 && pip uninstall -y torchtext
```
## 需下載的checkpoint
checkpoint_inpainting_text_image.bin
gligen-generation-text-box.bin

修改GLIGEN github repo下來的gligen_inference.py讓它可以跑我們要的內容，並將生成的圖片存到generated_sample這個資料夾，裡面又分成text_prompt1, text_prompt2, text+image
在472~496行這邊，可以去更改要跑的內容，（text_prompt1, text_prompt2, text+image）
生成圖片後接著先用一個./pytorch-fid/resize.py將原始140張照片resize成512*512
跑
```
$python -m pytorch_fid ./resized_picture ../GLIGEN/generation_samples/text_prompt1
$python -m pytorch_fid ./resized_picture ../GLIGEN/generation_samples/text_prompt2
$python -m pytorch_fid ./resized_picture ../GLIGEN/generation_samples/text+image
```

## DINO
## 環境跟HW1一樣
到training之前都跟github上一樣
環境:
Pytorch版本:conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
python=3.8
yapf=0.40.1
numpy 1.23.5
num_work=10->2
check_point model:DINO-4scale (checkpoint0011_4scale.pth)(12epoch)

先將生成的140張圖片的檔名用CVPDL3/change.py改成_generated.jpg之後，在將生成的圖片140張與origintrain2017new(140張原始圖片)合併起來放到DINO/COCODIR裡面並且在
要使用時將檔名改成train2017，並且修改json檔符合格式，要執行時，一樣將檔名改成instances_train2017.json
{這裡我將當前未使用到的training圖片資料夾各自命名為train2017old(原始), train2017prompt1(text grounding生成的), train2017prompt2(text grounding生成的), train2017image(image grounding生成的)，而json檔的命名原則一樣}

最後依照要執行的training修改DINO/train.bash



