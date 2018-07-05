# Quantized-CapsNet

## How to use

```
$ git clone https://github.com/Yusuke-Ikezaki/Quantized-CapsNet.git
```

### Setup

```
$ cd Quantized-CapsNet
$ python download_data.py
```

### Train

```
$ python main.py
```

### Test

```
$ python main.py --is_training=False --bits=4
```

## Result

### Parameters
Epoch: 50  
Batch size: 100

### Quantize only the weights of routing

| Bit | Accuracy |
| --- | -------- |
| 32  | 0.9945   |
| 8   | 0.9945   |
| 7   | 0.9946   |
| 6   | 0.9945   |
| 5   | 0.9945   |
| 4   | 0.9943   |
| 3   | 0.9931   |
| 2   | 0.9865   |
| 1   | 0.6376   |

### Quantize the weights and biases of routing

| Bit | Accuracy |
| --- | -------- |
| 32  | 0.9945   |
| 8   | 0.9945   |
| 7   | 0.9946   |
| 6   | 0.9947   |
| 5   | 0.9947   |
| 4   | 0.9947   |
| 3   | 0.9928   |
| 2   | 0.9917   |
| 1   | 0.6205   |

### Quantize all parameters of CapsNet  
Coming soon
