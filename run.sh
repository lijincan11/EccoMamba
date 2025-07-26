
python /home/ljc/source/EccoMamba/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /home/ljc/source/outputs/EccoMamba/ISIC2018_endoer+decoer/
python /home/ljc/source/EccoMamba/train.py --datasets_name isic2017 --epochs 500 --batch_size 24 --work_dir /home/ljc/source/outputs/EccoMamba/ISIC2017/

python /home/ljc/source/EccoMamba2/train.py --datasets_name isic2018 --epochs 500 --batch_size 24 --work_dir /home/ljc/source/outputs/EccoMamba/ISIC2018_decoer/

python /home/ljc/source/EccoMamba2/train.py --datasets_name isic2017 --epochs 500 --batch_size 24 --work_dir /home/ljc/source/outputs/EccoMamba/ISIC2017_decoer/

python /home/ljc/source/EccoMamba/test.py --datasets_name isic2017 --epochs 500 --batch_size 24 --work_dir /home/ljc/source/outputs/VMUnet/isic2017/ --best_model_path /home/ljc/source/outputs/VMUnet/isic2017/checkpoints/epoch160-miou0.7858-dsc0.8800.pth


python /home/ljc/source/EccoMamba/train.py --datasets_name isic2017 --epochs 300 --batch_size 24 --work_dir /home/ljc/source/outputs/VMUnet/isic2017/