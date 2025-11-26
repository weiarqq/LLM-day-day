## torch.outer

`torch.outer` 是 PyTorch 中的一个函数，用于计算两个向量的外积（outer product）。外积是一个数学运算，给定两个向量，它会生成一个矩阵。

**函数定义**

```python
torch.outer(input, vec2, *, out=None) → Tensor
```

**参数说明** 

- **input** (Tensor): 第一个一维输入向量（形状为 `n`）。
- **vec2** (Tensor): 第二个一维输入向量（形状为 `m`）。
- **out** (Tensor, optional): 输出张量（可选）。

**返回值**

返回一个形状为 `(n, m)` 的矩阵，其中第 `(i, j)` 个元素是 `input[i] * vec2[j]`。

**数学解释**

给定两个向量：

- `a = [a1, a2, ..., an]`
- `b = [b1, b2, ..., bm]`

外积的结果是一个矩阵 `C`，其中：

```
C[i][j] = a[i] * b[j]
```

**示例代码**

```
import torch

# 定义两个向量
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 计算外积
result = torch.outer(a, b)
print(result)
```

**输出**

```
tensor([[ 4,  5,  6],
        [ 8, 10, 12],
        [12, 15, 18]])
```

**解释**

- 第一行：`1 * [4, 5, 6] = [4, 5, 6]`
- 第二行：`2 * [4, 5, 6] = [8, 10, 12]`
- 第三行：`3 * [4, 5, 6] = [12, 15, 18]`

**注意事项**

- 输入必须是 **一维张量**（向量），否则会报错。
- 如果输入是空张量，结果也是空张量。
- 支持广播（broadcasting）语义，但通常用于明确的一维向量。

**替代方法**

如果 `torch.outer` 不可用，可以用以下方式实现：

```
result = a.view(-1, 1) * b.view(1, -1)
```

**总结**

`torch.outer` 是一个简洁高效的工具，用于计算两个向量的外积，生成一个矩阵。它在科学计算和机器学习中有广泛的应用。



## tensor.flatten 

`tensor.flatten()` 是 PyTorch 中的一个方法，用于**将任意形状的张量展平（拉平）成一维张量**（即向量）。它的作用是**保持所有元素不变，但去除所有维度，仅保留一个维度**。

**作用**

- 输入：任意形状的张量（如 (3, 2)、(2, 3, 4) 等）。
- 输出：一个一维张量（形状为 (n,)），包含输入张量的所有元素，顺序不变（按存储顺序展开）。

**示例**

1. 展平 2D 张量

```
import torch

x = torch.tensor([[1, 2], [3, 4]])  # shape (2, 2)
y = x.flatten()
print(y)
# 输出: tensor([1, 2, 3, 4])  # shape (4,)
```

2. 展平 3D 张量

```
x = torch.rand(2, 2, 3)  # shape (2, 2, 3)
y = x.flatten()
print(y.shape)  # 输出: torch.Size([12]) (因为 2 * 2 * 3=12)
```

**关键点**

1. 默认行为：

   - 从第 `0` 维开始展平，直到最后一维。
   - 等价于 `tensor.reshape(-1)` 或 `tensor.view(-1)`。

2. 指定维度范围展平：

   - 通过 `start_dim` 和 `end_dim` 参数可以控制从哪一维开始展平：

   ```
   x = torch.rand(2, 3, 4)
   y = x.flatten(start_dim=1)  # 只展平第1维和第2维，保留第0维
   print(y.shape)  # 输出: torch.Size([2, 12]) (3 * 4=12)
   ```

3. 非连续内存问题：

   - 如果输入张量不是内存连续的（如转置后的张量），`flatten()` 会返回一个**拷贝**（copy），而非视图（view）。此时建议先调用 `.contiguous()`。

**与类似方法的对比**

| 方法          | 作用           | 是否共享内存             |
| ------------- | -------------- | :----------------------- |
| `flatten()`   | 展平为一维张量 | 可能拷贝                 |
| `reshape(-1)` | 展平为一维张量 | 尽量共享内存             |
| `view(-1)`    | 展平为一维张量 | 必须共享内存（否则报错） |

**典型用途**

1. 全连接层输入：

   ```
   # 将卷积层的输出展平后输入全连接层
   conv_output = torch.rand(32, 64, 7, 7)  # batch=32, channels=64, size=7x7
   flattened = conv_output.flatten(1)      # shape (32, 64 * 7 * 7)
   fc_layer = nn.Linear(64 * 7 * 7, 1024)
   ```

2. 数据预处理：

   ```
   # 将图像数据展平为向量
   image = torch.rand(3, 256, 256)
   vector = image.flatten()  # shape (3 * 256 * 256,)
   ```

3. 计算统计量：

   ```
   # 计算所有元素的均值
   mean_value = x.flatten().mean()
   ```

**总结**

- `flatten()` 是**将张量压缩为一维**的便捷方法，默认按内存顺序展开。
- 若需保留部分维度，可使用 `start_dim` 和 `end_dim` 参数。
- 在需要共享内存时，优先考虑 `reshape()` 或 `view()`，但需注意张量的连续性。



## torch.polar

`torch.polar` 是 PyTorch 中的一个函数，用于将极坐标（幅度和角度）转换为复数形式的笛卡尔坐标。具体来说，它接受两个张量作为输入：一个表示幅度（模），另一个表示角度（弧度），并返回对应的复数张量。

**函数定义**

```
torch.polar(abs, angle) → Tensor
```

- abs (Tensor): 幅度（模）张量，必须是实数且非负数。
- angle (Tensor): 角度（弧度）张量，必须与 `abs` 的形状相同。
- 返回值: 复数张量，形状与 `abs` 和 `angle` 相同。

**数学原理**

对于输入的幅度 r 和角度 θ，`torch.polar` 计算复数的笛卡尔形式：

```
z=r⋅(cosθ+isinθ)
```

等价于：

```
z=r⋅eiθ
```

**示例**

```
import torch

# 定义幅度和角度
abs = torch.tensor([1.0, 2.0])       # 幅度 [1, 2]
angle = torch.tensor([0.0, 3.1415])  # 角度 [0, π]（弧度）

# 转换为复数
complex_tensor = torch.polar(abs, angle)
print(complex_tensor)
```

输出：

```
tensor([ 1.0000+0.0000j, -2.0000+0.0000j])
```

解释：

- 第一个复数：1⋅(cos0+isin0)=1+0i
- 第二个复数：2⋅(cosπ+isinπ)≈−2+0i

**主要用途**

1. **信号处理**：将极坐标表示的频谱转换为复数形式（如 FFT 结果）。
2. **图形学**：处理极坐标下的向量或位置。
3. **物理学/工程**：转换波动、电磁场等极坐标描述的物理量。

**注意事项**

- 输入 `abs` 必须是非负实数张量，否则会报错。
- `angle` 的单位是弧度，不是度数。
- 输出张量的 `dtype` 是复数类型（如 `torch.complex64` 或 `torch.complex128`）。

如果需要反向操作（从复数提取幅度和角度），可以使用 `torch.abs()` 和 `torch.angle()`。



## tensor.flatten

`tensor.flatten()` 是 PyTorch 中的一个方法，用于**将任意形状的张量展平（拉平）成一维张量**（即向量）。它的作用是**保持所有元素不变，但去除所有维度，仅保留一个维度**。

**作用**

- **输入**：任意形状的张量（如 `(3, 2)`、`(2, 3, 4)` 等）。
- **输出**：一个**一维张量**（形状为 `(n,)`），包含输入张量的所有元素，顺序不变（按存储顺序展开）。

**示例**

1. 展平 2D 张量

```
import torch

x = torch.tensor([[1, 2], [3, 4]])  # shape (2, 2)
y = x.flatten()
print(y)
# 输出: tensor([1, 2, 3, 4])  # shape (4,)
```

2. 展平 3D 张量

```
x = torch.rand(2, 2, 3)  # shape (2, 2, 3)
y = x.flatten()
print(y.shape)  # 输出: torch.Size([12]) (因为 2 * 2 * 3=12)
```

**关键点**

1. 默认行为：

   - 从第 `0` 维开始展平，直到最后一维。
   - 等价于 `tensor.reshape(-1)` 或 `tensor.view(-1)`。

2. 指定维度范围展平：

   - 通过 `start_dim` 和 `end_dim` 参数可以控制从哪一维开始展平：

   ```
   x = torch.rand(2, 3, 4)
   y = x.flatten(start_dim=1)  # 只展平第1维和第2维，保留第0维
   print(y.shape)  # 输出: torch.Size([2, 12]) (3 * 4=12)
   ```

3. 非连续内存问题：

   - 如果输入张量不是内存连续的（如转置后的张量），`flatten()` 会返回一个**拷贝**（copy），而非视图（view）。此时建议先调用 `.contiguous()`。

**与类似方法的对比**

| 方法          | 作用           | 是否共享内存             |
| ------------- | -------------- | ------------------------ |
| `flatten()`   | 展平为一维张量 | 可能拷贝                 |
| `reshape(-1)` | 展平为一维张量 | 尽量共享内存             |
| `view(-1)`    | 展平为一维张量 | 必须共享内存（否则报错） |

**典型用途**

1. 全连接层输入：

   ```
   # 将卷积层的输出展平后输入全连接层
   conv_output = torch.rand(32, 64, 7, 7)  # batch=32, channels=64, size=7x7
   flattened = conv_output.flatten(1)      # shape (32, 64 * 7 * 7)
   fc_layer = nn.Linear(64 * 7 * 7, 1024)
   ```

2. 数据预处理：

   ```
   # 将图像数据展平为向量
   image = torch.rand(3, 256, 256)
   vector = image.flatten()  # shape (3 * 256 * 256,)
   ```

3. 计算统计量：

   ```
   # 计算所有元素的均值
   mean_value = x.flatten().mean()
   ```

总结

- `flatten()` 是**将张量压缩为一维**的便捷方法，默认按内存顺序展开。
- 若需保留部分维度，可使用 `start_dim` 和 `end_dim` 参数。
- 在需要共享内存时，优先考虑 `reshape()` 或 `view()`，但需注意张量的连续性。



## torch.einsum

`torch.einsum` 是 PyTorch 中用于执行 **爱因斯坦求和约定** 的函数，它提供了一种简洁而强大的方式来表达多维张量之间的复杂运算（如矩阵乘法、转置、求和等）。其核心语法为：

```
torch.einsum('subscripts', *tensors)
```

### 📖 核心规则

1. **下标字符串**：用逗号分隔输入张量的维度标签，箭头 `->` 后指定输出格式。
2. **重复标签**：在输入中重复出现的标签表示该维度需要**求和消去**。
3. **输出标签**：仅出现在输出中的标签会被保留，未指定的维度会被求和。

------

### 🧩 常见用法示例

#### 1. **矩阵乘法**（等价于 `A @ B`）

```
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = torch.einsum('ik,kj->ij', A, B)  # 结果形状: (2, 4)
```

#### 2. **向量点积**（等价于 `torch.dot(a, b)`）

```
a = torch.randn(5)
b = torch.randn(5)
dot = torch.einsum('i,i->', a, b)  # 结果: 标量
```

#### 3. **逐元素乘法求和**（等价于 `(A * B).sum()`）

```
A = torch.randn(3, 3)
B = torch.randn(3, 3)
summed = torch.einsum('ij,ij->', A, B)
```

#### 4. **矩阵转置**（等价于 `A.T`）

```
A = torch.randn(3, 4)
AT = torch.einsum('ij->ji', A)  # 形状变为 (4, 3)
```

#### 5. **批量矩阵乘法**

```
batch_A = torch.randn(5, 2, 3)  # 批量大小=5
batch_B = torch.randn(5, 3, 4)
result = torch.einsum('bij,bjk->bik', batch_A, batch_B)  # 结果形状: (5, 2, 4)
```

#### 6. **高阶张量缩并**

```
X = torch.randn(2, 3, 4)
Y = torch.randn(4, 5, 6)
Z = torch.einsum('abc,cde->abde', X, Y)  # 结果形状: (2, 3, 5, 6)
```

------

### ⚠️ 使用技巧与注意事项

1. **省略求和维度**：不写入输出标签的维度会被自动求和： `# 对矩阵所有元素求和 total = torch.einsum('ij->', A)  # 等价于 A.sum()`
2. **广播支持**：隐式支持广播规则： `A = torch.randn(3, 1, 4) B = torch.randn(2, 4) C = torch.einsum('abc,cd->abd', A, B)  # 结果形状: (3, 2, 4)`
3. **性能优化**：对于常见操作（如矩阵乘），直接使用 `torch.matmul()` 可能更快，但 `einsum` 更灵活。
4. **调试提示**：若出现维度不匹配错误，检查： 输入张量维度是否与下标匹配 求和维度的大小是否一致

------

### 🌟 进阶示例

#### 双线性变换：

```
U = torch.randn(10, 20)
V = torch.randn(10, 30)
W = torch.randn(20, 30)
output = torch.einsum('ik,jk,ij->', U, V, W)  # 结果: 标量
```

#### 迹运算（Trace）：

```
matrix = torch.randn(4, 4)
trace = torch.einsum('ii->', matrix)  # 等价于 torch.trace(matrix)
```

------

掌握 `torch.einsum` 能极大简化复杂张量操作代码，建议结合实际问题多加练习！