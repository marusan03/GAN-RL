# GAN-RL


# PROGRESS
[ ] GDMのforward作成
    [x] forwardの作成
    [ ] loss関数の作成
    [ ] test

[ ] Discriminator作成
    [x] modelの作成
    [ ] loss関数の作成
    [ ] test

[ ] Reward Predictor作成
    [x] modelの作成
    [ ] loss関数の作成
    [ ] test

[ ] DQNの作成
    [ ] forwardの作成
    [ ] Repray Memoryの作成
    [ ] target Q networkの作成
    [ ] test

[ ] OpenAI gymの作成
    [ ] 環境はどうするか
    [ ] トレーニング用に作成

[ ] trainの作成
    [ ] バッチノーマライゼーションの設定
    [ ] それぞれの学習の設定
    [ ] チェックポイントの作成
        [ ] 各モデルごとに作成
        [ ] 読み込みも

[ ] testの作成
    [ ] チェックポイントの読み込み
    [ ] ログの作成
    [ ] グラフの作成
    [ ] 動画の作成

[ ] extra
    [ ] できればA3Cや他の深層強化学習と組み合わせたい
    [ ] RPの仕組みをDiscriminatorに組み込みたい(like ACGAN)
    [ ] GDMをもっと先まで予測できるようにしたい

# REFERENCE
- SN-Convolution
https://github.com/minhnhat93/tf-SNDCGAN/blob/master/libs/sn.py
- DQN
https://github.com/devsisters/DQN-tensorflow
- GDM
https://arxiv.org/abs/1806.05780