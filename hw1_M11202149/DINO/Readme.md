到training之前都跟github上一樣
環境:
Ubuntu 20.04
Pytorch版本:conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
python=3.8
yapf=0.40.1
numpy 1.23.5
num_work=10->2
check_point model:DINO-4scale (12epoch)


位置:dataset-->COCODIR(annotation(instances_train2017.json,instances_val2017.json),test,train2017,val2017)
training-->sh train.bash
train 完之後資料存在output檔
要執行test圖片預測用Python testvisual.py(換照片)
要產生output.json，執行python testwrite.py





