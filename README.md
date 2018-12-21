# GAN-RL


# PROGRESS
- GDMのforward作成  
    [x] forwardの作成  
    [x] loss関数の作成  
    [x] test  

- Discriminator作成
    [x] modelの作成  
    [x] loss関数の作成  
    [x] test  

- Reward Predictor作成  
    [x] modelの作成  
    [x] loss関数の作成  
    [ ] test  
        - うまく動いていない？  
        - lossが振動する  

- DQNの作成
    - [x] forwardの作成  
    - [x] Repray Memoryの作成  
    - [x] target Q networkの作成  
    - [x] test  

- OpenAI gymの作成
    - [x] 環境はどうするか  
        - Pongを使用
    - [x] トレーニング用に作成  

- trainの作成
    - [x] バッチノーマライゼーションの設定  
    - [x] それぞれの学習の設定  
    - [x] チェックポイントの作成  
        - [x] 各モデルごとに作成  
        - [x] 読み込みも  

- testの作成  
    - [x] チェックポイントの読み込み  
    - [x] ログの作成  
    - [x] グラフの作成  
    - [ ] 動画の作成  

- extra  
    - できればA3Cや他の深層強化学習と組み合わせたい  
    - RPの仕組みをDiscriminatorに組み込みたい(like ACGAN)  
    - GDMをもっと先まで予測できるようにしたい  

# REFERENCE
- SN-Convolution
https://github.com/minhnhat93/tf-SNDCGAN/blob/master/libs/sn.py
- WGAN-GP
https://github.com/igul222/improved_wgan_training
- DQN
https://github.com/devsisters/DQN-tensorflow
- GATS
https://arxiv.org/abs/1806.05780
https://github.com/kazizzad/GATS