# 故障排除指南

## "Too many open files" 错误

### 问题描述

在处理大量数据时，可能会遇到以下错误：
```
Unexpected error: [Errno 24] Too many open files
```

这是因为系统同时打开的文件句柄数量超过了系统限制。

### 原因分析

1. **文件句柄未及时关闭**: 在高并发处理时，如果图片文件句柄没有及时关闭，会累积导致超出限制
2. **系统限制过低**: 默认的文件描述符限制可能不足以支持高并发处理
3. **并发数过高**: 如果设置的worker数量过高，会同时打开大量文件

### 解决方案

#### 1. 检查当前系统限制

运行检查工具：
```bash
python utils/check_file_limits.py
```

#### 2. 临时提高限制（当前会话）

```bash
# 查看当前限制
ulimit -n

# 临时提高到8192（需要权限）
ulimit -n 8192
```

#### 3. 永久提高限制（需要root权限）

编辑 `/etc/security/limits.conf`，添加：
```
* soft nofile 8192
* hard nofile 16384
```

然后重新登录或重启。

#### 4. 在代码中提高限制

如果程序有权限，可以在代码开始时提高限制：

```python
import resource

# 获取当前硬限制
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

# 提高软限制（不超过硬限制）
new_soft = min(8192, hard)
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
```

#### 5. 降低并发数

如果无法提高系统限制，可以降低并发数：

```bash
# 使用更少的worker
python main.py --json data.json --workers 2

# 或者禁用并发
python main.py --json data.json --no-concurrent
```

#### 6. 使用多线程而非多进程

多进程会为每个进程打开文件，消耗更多文件句柄。如果可能，使用多线程：

```bash
# 默认使用多线程（不指定 --multiprocessing）
python main.py --json data.json --workers 4
```

### 已修复的问题

代码已经修复了以下问题：

1. **图片文件句柄管理**: 
   - `_load_image()` 方法现在会立即读取文件内容并关闭文件句柄
   - `_encode_image()` 方法在使用完图片后显式关闭图片对象

2. **资源清理**:
   - 使用 `try-finally` 确保资源被正确释放
   - 在并发处理中定期进行垃圾回收

### 建议的配置

对于大规模数据处理：

1. **系统限制**: 至少 8192 个文件描述符
2. **并发数**: 根据系统限制调整，建议：
   - 文件限制 4096: 使用 2-4 个worker
   - 文件限制 8192: 使用 4-8 个worker
   - 文件限制 16384: 使用 8-16 个worker
3. **处理模式**: 
   - 小规模（<1000条）: 可以使用多进程
   - 大规模（>10000条）: 建议使用多线程或降低并发数

### 验证修复

修复后，重新运行处理：

```bash
# 检查系统限制
python utils/check_file_limits.py

# 使用较低的并发数重新处理
python main.py --json data.json --workers 2 --output new_output.json

# 检查结果
python utils/sample_results.py new_output.json sample_check.json -n 10
```

如果仍有问题，可以：
1. 进一步降低并发数
2. 增加系统文件描述符限制
3. 使用串行处理（`--no-concurrent`）

