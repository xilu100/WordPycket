yhege1wen# WordPycket

WordPycket 是一个本地运行的 Qt 桌面词汇学习工具。它可以从 CSV 导入词库，也可以从 PDF 自动解析词频并生成 CSV；学习进度、例句和词库数据都保存在项目目录内。

## 普通用户使用

最终用户不需要安装 Python、conda、pip、spaCy、NLTK 或 CUDA Toolkit。

### Windows

如果拿到的是发布版目录，双击：

```text
run.exe
```

如果是直接从源码仓库使用，双击：

```text
run.bat
```

`run.bat` 不会寻找或使用电脑上已有的 Python。它会先在项目目录内下载 `uv.exe`，再把完整 Python 3.11 安装到 `runtime/python/`，然后创建 `.venv/`、`model/` 和 `data/`，并自动配置环境后启动。检测到 NVIDIA 显卡时会要求安装 CUDA 版 `llama-cpp-python`；CUDA 版装不上会停止并显示错误，避免悄悄退到 CPU 版。

### macOS

如果是直接从源码仓库使用，双击：

```text
run.command
```

如果系统提示无法打开，先在终端进入项目目录执行一次：

```bash
chmod +x run.command
```

`run.command` 不会寻找或使用电脑上已有的 Python。它会先在项目目录内下载 `uv`，再把完整 Python 3.11 安装到 `runtime/python/`，然后创建 `.venv/`、`model/` 和 `data/`，并自动配置环境后启动。Apple Silicon 机器会优先配置 Metal 版 `llama-cpp-python`；其它机器使用 CPU。

### 本地文件

所有运行期文件都放在项目目录内：

```text
.venv/
runtime/
model/
input/
data/
```

删除整个项目文件夹后，不会留下 WordPycket 的数据库、模型或缓存。

## 功能

- 主页提供 `学习`、`复习`、`词表` 三个入口。
- `学习` 页面只显示学习池中的单词卡片和操作按钮。
- `复习` 页面只显示复习池中的单词卡片和操作按钮。
- `词表` 页面显示当前 CSV 的词库，支持通过管理弹窗选择/删除 CSV、上传 CSV、上传 PDF、搜索、查看状态和删除选中词条。
- 智能补充/修正使用 `model/` 目录中的本地 `.gguf` 模型。

## 数据

- 启动时读取 `input/` 中当前选中的 CSV；`input/` 可以同时保存多个 CSV。
- 上传 CSV 会先校验列名，通过后复制到 `input/` 并切换到该 CSV。上传 PDF 会自动解析并在 `input/` 中生成一个 CSV。
- 每个 CSV 使用独立 SQLite 数据库保存学习进度和例句；切换 CSV 不会丢失其它 CSV 的数据库。删除 CSV 前会提示确认，确认后对应数据库会一起删除。
- CSV 会自动检测列名语言，但列结构必须是固定的 5 列。例如英语是 `Index`、`English`、`Chinese`、`Frequency`、`Forms`；德语是 `Index`、`Deutsch`、`Chinesisch`、`Häufigkeit`、`Formen`。
- 上传 PDF 会提取文本、自动识别语言、统计词频和固定词组频率，并生成同一套固定列结构的 CSV。扫描图片 PDF 需要先 OCR 成可复制文本。
- PDF 词形归并强制使用语言库：英文需要 `nltk` 的 `wordnet` 数据，其它拉丁字母语言需要对应 spaCy 模型；缺少语言库时会提示安装。
- SQLite 保存词条、例句、会/不会次数、复习时间和学习状态。
- 正常启动会按 CSV 更新单词释义、频率和词形，不会清空已有例句或学习进度。
- `重置学习进度` 会保留 `Index`、英文、中文、频率、词形、例句和例句中文；其它学习进度字段会清零。

## 学习规则

- `学习池`：`会` 次数小于 5。
- `复习池`：`会` 次数大于等于 5，且还未到已学习时间。
- `已学习池`：复习池中继续累计 3 次 `会` 后，等待 24 小时，到时自动进入。
- 点击 `不会`：`不会` 次数加 1。
- 点击 `会`：`会` 次数加 1；累计到 5 次后进入复习池。
- 点击 `绝对会`：直接把 `会` 次数提升到至少 5，并进入复习池。
- 复习池中的词达到 `会` 次数 8 时，会记录 24 小时后的已学习时间；到时间前仍属于复习池。

## 抽取规则

- 新词使用加权随机抽取，不是固定顺序。
- 权重公式为 `(不会次数 + 1) / (会次数 + 1)`。
- `会` 次数越少、`不会` 次数越多，越容易被抽到。
- 每次进入 `学习` 或 `复习` 会开启一次会话。
- 同一会话中，每个新词最多显示一次。
- 上一会话出现过的词会在下一会话跳过，到下下次会话重新加入抽取池。

## 历史规则

- 只有点击过 `不会`、`会` 或 `绝对会` 的词会进入本次会话历史。
- 历史词按操作顺序保存，可用 `上一个` / `下一个` 顺序浏览。
- 浏览历史词时不显示 `不会`、`会`、`绝对会`，避免重复计数。
- 在历史末尾点击 `下一个`，会继续抽取本次会话中还没有显示过的新词。

## 本地模型

- 智能补充/修正使用 `model/` 目录中的 `.gguf` 模型。
- `model/` 目录中只能存在一个 `.gguf` 文件；如果存在多个，智能功能会停止并提示清理。
- 如果 `model/` 中没有 `.gguf` 文件，首次使用智能功能时会自动从 Hugging Face 下载默认模型：
  `Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf`。
- 如果用户自行放入 `.gguf` 模型，软件会优先使用用户模型，并在智能功能启动前提示兼容性不保证。
- 默认模型来源：Hugging Face `Qwen/Qwen2.5-3B-Instruct-GGUF`。

## 维护者构建

以下步骤只给维护者/发布者使用，普通用户不需要执行。

先创建构建环境：

```powershell
python scripts/setup_env.py
```

脚本会自动检测 NVIDIA CUDA、macOS Apple Metal/MPS 和 CPU。检测到 NVIDIA 显卡时会安装 `llama-cpp-python` 的预编译 CUDA wheel；没有匹配 wheel 时会停止并报错，避免把有 NVIDIA 的机器配置成 CPU 版。没有可用 GPU/驱动时才会使用 CPU；项目不能在没有 NVIDIA GPU 和驱动的机器上凭空提供可用 CUDA。

只安装部分 spaCy 模型可用：

```powershell
python scripts/setup_env.py --spacy-models en,de
```

强制 CUDA 或 Metal，失败时不回退：

```powershell
python scripts/setup_env.py --device cuda --strict-accel
python scripts/setup_env.py --device mps --strict-accel
```

构建便携版：

```powershell
.\.venv\Scripts\python.exe scripts\build_portable.py --name run --include-input --include-model
```

macOS 上使用：

```bash
./.venv/bin/python scripts/build_portable.py --name run --include-input --include-model
```

也可以直接一键创建/更新构建环境并打包：

```powershell
python scripts/release_portable.py --name run
```

构建完成后分发整个目录，不要只发 exe：

```text
dist/run/
```

Windows 用户运行：

```text
dist/run/run.exe
```

macOS 用户需要在 macOS 上构建后运行对应的 `dist/run/run` 或 `.app` 包；Windows 构建物不能替代 macOS Metal 版本。

## 手动开发安装

一般不需要，除非你在调试依赖。

Windows：

```powershell
pip install -e .
python -m nltk.downloader wordnet omw-1.4
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

macOS：

```bash
pip install -e .
python -m nltk.downloader wordnet omw-1.4
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

维护者本地直接运行：

```powershell
.\.venv\Scripts\python.exe -m wordpycket.main
```

macOS：

```bash
./.venv/bin/python -m wordpycket.main
```

## 架构

- `domain`：领域实体和仓储接口，不依赖外层实现。
- `application`：应用服务，编排用例。
- `infrastructure`：SQLite 仓储实现、CSV 导入、PDF 解析和本地模型调用。
- `presentation`：Qt GUI 展示层。

依赖方向为：`presentation -> application -> domain`，`infrastructure -> domain`。
