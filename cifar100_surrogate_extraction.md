# 用 CIFAR100 當 Surrogate 做 Model Extraction

> 給專題生的講義。範圍只到「使用外部資料集當 surrogate」這一步；完全不依賴外部資料的 generator-based DFME 是另一個主題。

---

## 1. 我們在做什麼

**場景：** 有一個受害者模型（victim / teacher）— CIFAR10 上訓練好的 ResNet18，已經達到 94.81% 準確率。攻擊者只能用「query API」存取它：丟一張 input，拿回 10 維的 logits（或 softmax 後的 soft label）。攻擊者**沒有** teacher 的權重、沒有 training script、也沒有 CIFAR10 訓練集。

**目標：** 訓練一個自己的 student 模型，行為盡量接近 teacher。理想結果是 student 對任何 input 都跟 teacher 給出一樣的預測。

**為什麼有人會做這件事：**
- 偷別人辛苦訓練的商業模型（盜版 API）
- 把雲端模型「下載」到本地後做對抗攻擊（white-box adversarial example，攻擊成功率比 black-box 高很多）
- 規避 API 的 rate-limit / 收費

---

## 2. 為什麼叫 "data-free"

嚴格的 data-free 是「攻擊者完全不用任何外部資料集」（要靠 generator 自己合成 query）。

我們今天放鬆這個假設，叫**「surrogate-data extraction」**：攻擊者拿不到 teacher 的訓練資料 (CIFAR10)，但可以用其他公開資料集當 query。這在實務上更現實 — 公開資料集網路上一抓一大把，而且效果好很多（後面會看到）。

---

## 3. Surrogate dataset 怎麼選

**核心原則：surrogate 的影像分布要跟 teacher 訓練資料夠接近**，這樣 teacher 對它的 response 才有意義。

| 候選 | 解析度 | 通道 | 跟 CIFAR10 的距離 |
|---|---|---|---|
| MNIST | 28×28 | 1 (灰階) | 遠：手寫數字 vs. 自然影像 |
| **CIFAR100** | **32×32** | **3** | **近：同樣是自然影像，class 完全 disjoint** |
| TinyImageNet | 64×64 (resize 32) | 3 | 近：natural image |
| SVHN | 32×32 | 3 | 中：街景數字，物體多樣性低 |

**CIFAR100 的優勢：**
1. 解析度與通道數完全匹配 → 不用做 resize/灰階轉 RGB 之類的手腳
2. 類別完全 disjoint（CIFAR100 沒有 airplane / dog / cat / ... 這些 CIFAR10 類別）→ 攻擊者真的沒看過 CIFAR10 標籤
3. Low-level 影像統計（紋理、色彩、空間頻率）與 CIFAR10 幾乎一致

---

## 4. 演算法（Knowledge Distillation 變體）

### 4.1 想法

對每張 CIFAR100 image x：
- 送給 teacher，拿到 logits T(x)
- 送給 student，拿到 logits S(x)
- Loss 把兩者的 softmax 分布拉近 → student 學會「在這張 x 上，teacher 會怎麼想」

### 4.2 Loss：KL divergence with temperature

```
L_KD(x) = T² · KL( softmax(T(x)/T) || softmax(S(x)/T) )
```

`T` 是 temperature（這裡用 T=4），溫度越高 softmax 越平、能傳遞更多「次優類別」的相對機率資訊（dark knowledge）。乘 T² 是為了讓梯度量級在不同 T 下保持可比。

只用「teacher 的 argmax 是哪一類」這種 hard label 也可以，但 soft label 含的資訊量大很多。例如 teacher 看到一張 CIFAR100 的「狼」可能輸出 `dog: 0.6, cat: 0.2, deer: 0.1, ...`，student 學到的不只是「分類為 dog」，還包括「dog 跟 cat 在這張圖上很接近」這個結構化知識。

### 4.3 Pseudocode

```python
for epoch in range(epochs):
    for x in cifar100_loader:                  # 我們的 query 集
        with torch.no_grad():
            t_logits = teacher(x)              # API 呼叫 / 黑盒推論
        s_logits = student(x)
        loss = kd_loss(s_logits, t_logits, T=4)
        loss.backward()
        optimizer.step()
```

---

## 5. 實作關鍵點

### 5.1 Input alignment：用 CIFAR10 的 normalization

Teacher 訓練時看到的是 `(image - CIFAR10_mean) / CIFAR10_std` 後的 input。攻擊者送 CIFAR100 image 進去 query 時，**也要用 CIFAR10 的 mean/std normalize**，不是 CIFAR100 自己的統計值。

理由：teacher 期待的 input 分布是「CIFAR10 normalized 後的樣子」。如果用 CIFAR100 stats 去 normalize，input 會落在 teacher 沒看過的區域，logits 變得不可靠。

[extract.py:20-21](extract.py#L20-L21):
```python
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
```

### 5.2 Query data augmentation

對 CIFAR100 query 做 RandomCrop + HFlip ([extract.py:24-40](extract.py#L24-L40)) — 這是 distillation 領域的 trick：用 augmentation 創造更多「teacher 沒看過的 query」，等於擴大有效 query 預算，能平滑 student 的 decision boundary。

### 5.3 Teacher 不需要梯度

```python
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
```

加上 `with torch.no_grad():` 包住 teacher forward，可以省下大量 activation memory（teacher 只是「query API」，不需要 backprop）。

### 5.4 KD loss 對應實作

[extract.py:80-86](extract.py#L80-L86)：
```python
def kd_loss(student_logits, teacher_logits, temperature):
    T = temperature
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean',
    ) * (T * T)
```

注意 PyTorch 的 `F.kl_div` 第一個參數要 **log_softmax**（不是 softmax），第二個是 softmax。順序錯掉很常見，會 train 出垃圾。

---

## 6. 評估指標：accuracy 與 agreement

兩個指標**參考點不同**：

| 指標 | 比對對象 | 衡量什麼 |
|---|---|---|
| Accuracy | 真實 label | Student 在 CIFAR10 task 上的絕對能力 |
| **Agreement** | **Teacher 的預測** | **Student 多忠實地複製 teacher（fidelity）** |

Model extraction 的主要指標是 **agreement**：
- 攻擊者目標是「複製模型行為」，包括 teacher 的錯誤
- Teacher 答錯（dog → cat）、student 跟著答錯 → agreement +1，accuracy +0
- 上限差異：accuracy 上限受 teacher 自己 acc 限制；agreement 上限是 100%（完美複製）

[extract.py:65-77](extract.py#L65-L77)：
```python
@torch.no_grad()
def agreement(student, teacher, loader, device):
    match = total = 0
    for x, _ in loader:
        x = x.to(device)
        s_pred = student(x).argmax(1)
        t_pred = teacher(x).argmax(1)
        match += (s_pred == t_pred).sum().item()
        total += x.size(0)
    return match / total
```

---

## 7. 實驗結果

| Surrogate | Student CIFAR10 acc | Agreement | Fidelity gap | 說明 |
|---|---|---|---|---|
| MNIST (28×28 灰階) | 22.39% | 22.41% | **72.42 pp** | 跨 domain 失敗 |
| **CIFAR100 (32×32 彩色)** | **91.86%** | **94.38%** | **2.95 pp** | 同 domain 成功 |

> Teacher CIFAR10 test acc: 94.81%
> 兩組設定都跑 50 epochs、batch 256、SGD lr=0.1 cosine、KD T=4

**只是換了 query 集**，student 的 fidelity 從 22% 飛到 94%。同樣的 KD loss、同樣的學生架構、同樣的 query 預算（50000 query images × 50 epochs）。

---

## 8. 為什麼差這麼多 — 核心觀念

### 8.1 MNIST 失敗的原因

MNIST 對 CIFAR10-pretrained teacher 來說是 **out-of-distribution (OOD)**：
- 灰階填空背景 vs. teacher 看慣的彩色自然影像
- 形狀只有手寫筆劃 vs. teacher 看慣的物體輪廓

Teacher 對 OOD input 的反應通常是「過度自信地塌縮到少數幾類」。例如所有 MNIST 數字可能都被 teacher 預測為 `automobile` 或 `truck`，logits 分布幾乎一樣。Student 從這種 query 學到的東西**完全沒辦法區分 CIFAR10 的 10 個類別**，因為 teacher 的回應不帶有「在 CIFAR10 中該怎麼分」的資訊。

### 8.2 CIFAR100 成功的原因

CIFAR100 image 對 teacher 來說是 **in-distribution**（同 domain natural image，只是新類別）：
- Teacher 對 CIFAR100 中的 `wolf` 可能回 `dog: 0.65, cat: 0.20, deer: 0.10, ...`
- 對 `bus` 可能回 `truck: 0.50, automobile: 0.30, ...`
- 對 `lion` 可能回 `cat: 0.55, dog: 0.30, ...`

這些 soft label 帶有 teacher 對 CIFAR10 類別之間相對關係的完整知識（哪些類別像、哪些不像）。Student 從這 50000 張 image × 50 epoch 的 query 中，等於拿到了 teacher 在 CIFAR10 整個輸入空間上的 decision boundary 樣本，自然能複製出來。

### 8.3 一句話總結

> Surrogate 是不是 in-distribution，是 model extraction 成功與否最關鍵的因素 — 比演算法、loss 設計、超參都重要。

---

## 9. 延伸討論

### 9.1 這還算 "data-free" 嗎？

技術定義很嚴：「不用任何外部資料集」。我們用了 CIFAR100，所以**嚴格說不算**，比較精確的講法是 "surrogate-data attack"。

業界比較鬆的用法：「不用 victim 的訓練資料」也叫 data-free。本實驗在這個寬鬆定義下成立。

真的完全 data-free 要走 generator-based DFME（Truong et al. 2021） — 用 GAN-like generator 從 noise 合成 query，跟 student 對抗訓練。**這是另一個主題**，下次講。

### 9.2 防禦面（research idea）

如果你是 teacher 的擁有者，想擋這種攻擊，可以做：
- **Output perturbation**：給 logits 加 noise（會傷自己 user 體驗）
- **Top-k truncation**：只回傳前 k 大的 prob，砍掉 dark knowledge
- **Temperature scaling 反向操作**：降低 logits 溫度，讓 soft label 接近 one-hot，少傳遞資訊
- **Watermark / fingerprinting**：在 model 裡植入隱藏 trigger，被偷走後可以驗證
- **Query-based detection**：偵測「短時間內大量奇怪 query」這種模式

每種方法都有 utility / security trade-off，是現在很活躍的 research 方向。

### 9.3 學生可以延伸的實驗

1. 把 surrogate 換成 SVHN / TinyImageNet，看 agreement 怎麼變
2. 改變 query budget（每個 epoch 用幾張 image），畫 budget vs agreement 曲線
3. 改變 KD temperature T = 1, 2, 4, 8, 16 看哪個最好
4. Hard label only（只有 argmax，沒有 soft logits）能 train 到什麼程度？— 對應 teacher 只 expose top-1 的場景
5. Teacher 跟 student 架構不同（如 teacher = ResNet18, student = ResNet50 / VGG）會怎樣？

---

## 10. 程式檔案地圖

| 檔案 | 用途 |
|---|---|
| [models/resnet_cifar.py](models/resnet_cifar.py) | CIFAR 風 ResNet18（首層 3×3, 沒 maxpool） |
| [train_teacher.py](train_teacher.py) | 在 CIFAR10 訓練 teacher（一次性，存到 checkpoints/） |
| [extract.py](extract.py) | **本主題：用 CIFAR100 query teacher，KD 訓練 student** |
| [eval.py](eval.py) | 比較 student / teacher 在 CIFAR10 test set 的 acc 與 agreement |
| `checkpoints/teacher_resnet18_cifar10.pth` | Teacher 權重（94.81%） |
| `checkpoints/student_resnet18_cifar100query.pth` | CIFAR100 surrogate 訓出來的 student（91.86%, agreement 94.38%） |
| `checkpoints/student_resnet18_mnistquery.pth` | MNIST baseline 對照組（22.39%, agreement 22.41%） |

### 重現實驗（在 dsem env 下）

```bash
# 1. 先 train teacher（一次就好）
python train_teacher.py --epochs 100

# 2. CIFAR100 surrogate 抽取
python extract.py --epochs 50 \
    --student-ckpt ./checkpoints/student_resnet18_cifar100query.pth

# 3. 評估
python eval.py \
    --student-ckpt ./checkpoints/student_resnet18_cifar100query.pth
```

---

## 11. 重點 takeaway（給你帶回家）

1. **Model extraction 不需要受害者的訓練資料就能做** — 一個 in-distribution 的公開 surrogate 就夠了
2. **KD with soft labels 是最樸素也最有效的攻擊框架** — 沒什麼花俏的 loss，就 KL divergence
3. **Surrogate 的 domain 是否匹配 teacher 是成敗關鍵** — 比 algorithm 還關鍵（CIFAR100 → 94%, MNIST → 22%）
4. **Agreement 才是攻擊指標** — accuracy 是「task transfer」，agreement 是「fidelity」，後者才是 extraction 的真正目標
5. **防禦端可以動 logits 介面**（perturbation, top-k, temperature）— 但都有 utility cost
