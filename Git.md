# NumPy MLP MNIST 项目协作说明

本项目使用 **Git + GitHub** 进行代码管理和小组协作。

本文档由AI生成[ac01].

项目目标：

- 只使用 Python + NumPy 实现多层感知机 MLP
- 在 MNIST 手写数字数据集上训练和测试
- 测试集准确率达到 95% 以上
- 完成实验报告，包括反向传播推导、训练曲线和超参数实验

---

## 1. Git 是什么？

Git 是一个版本管理工具。

你可以把它理解成：

> Git 会记录每一次代码修改，让我们可以知道谁改了什么，也可以在出错时回到以前的版本。

GitHub 是一个远程代码托管平台。

你可以把它理解成：

> GitHub 是放在网上的项目文件夹，所有组员都可以从上面下载代码、上传自己的修改、查看别人改了什么。

善用AI.

> 比如说，不知道如下载项目到一个已经存在的文件夹，问AI：我想把一个Github上的项目部署到本地同名文件夹，该如何做

---

## 2. 本项目的协作规则

为了避免代码互相覆盖，必须遵守以下规则。

### 2.1 不要直接修改 main 分支

`main` 分支只存放稳定版本。

禁止直接在 `main` 分支上写代码、提交代码、push 代码。

正确做法是：

1. 从 `main` 创建自己的功能分支（如何创建可以询问AI）
2. 在自己的分支上写代码
3. push 到 GitHub
4. 经其他组员检查后合并到 `main`

### 2.2 每个人负责自己的模块

每个人尽量只修改自己负责的文件，避免多人同时改同一个文件导致冲突。

---

## 3. 开发流程

每次开发一个新功能，都按照下面流程来。

注意：

- 组员可以把自己的分支 push 到 GitHub。
- 不要直接在 `main` 分支上写代码。
- 不要直接 push 到 `main`。
- 所有新功能都应该先在自己的 `feature/*` 分支上完成，再通过 Pull Request 合并到 `main`。

### Step 1：切换到 main 并更新本地代码

每次开始写代码前，先回到 `main` 分支，并从 GitHub 拉取最新代码：

```bash
git switch main
git pull origin main
```

这样可以保证你是基于最新版本开始开发，减少代码冲突。


### Step 2：创建自己的功能分支

不要直接在 `main` 上写代码。

创建一个新的功能分支：

```bash
git switch -c feature/你的功能名
```

例如，如果你要写优化器模块：

```bash
git switch -c feature/optimizer
```

分支名使用英文小写，用 `-` 连接单词。

重点是英文，并且能表达分支用途

### Step 3：在自己的分支上写代码

修改自己负责的文件。

### Step 4：查看修改状态

写完一部分功能后，先查看哪些文件被修改了：

```bash
git status
```

### Step 5：添加修改到暂存区

如果只想添加某个文件：

```bash
git add 文件名
```

如果确定所有修改都应该提交，可以使用：

```bash
git add .
```

### Step 6：提交修改

提交代码：

```bash
git commit -m "Implement SGD optimizer"
```

commit message 要清楚说明这次修改做了什么。

不要把很多不相关的修改放在同一个 commit 里。

好的 commit：

```bash
git commit -m "Implement ReLU forward and backward"
git commit -m "Add MNIST data loader"
git commit -m "Fix softmax numerical stability"
```

不好的 commit：

```bash
git commit -m "update"
git commit -m "fix"
git commit -m "change something"
git commit -m "finish all"
```

关于commit 的规范，有兴趣的可以问AI或者善用搜索引擎，我们不关心这个，能明确表达这次commit 干了什么就好了


### Step 7：上传自己的分支到 GitHub

第一次上传当前分支时，需要使用：

```bash
git push -u origin 分支名
```

例如：

```bash
git push -u origin feature/optimizer
```

之后如果继续在同一个分支上修改并提交，再上传时只需要：

```bash
git push
```

注意：

不要执行：

```bash
git push origin main
```

也不要在 `main` 分支上直接 push。


### Step 8：合并到main 分支（pull request）

上传分支后，打开 GitHub 仓库页面。

通常 GitHub 会自动显示：

```text
Compare & pull request
```

点击它，然后填写 Pull Request 说明。

如果没有自动显示，可以手动进入：

```text
Pull requests
→ New pull request
```

然后选择：

```text
base: main
compare: 你的功能分支
```

例如：

```text
base: main
compare: feature/optimizer
```

然后创建 Pull Request。


### Step 9：等待检查和合并

创建 Pull Request 后，至少让一位组员检查代码。

检查内容包括：

- 代码是否能运行
- 是否影响其他模块
- 是否有明显 bug
- 是否修改了不该修改的文件
- commit message 是否清楚

确认没有问题后，再把 Pull Request 合并到 `main`。

## 4. Pull Request 规范

微信群里说一声，讲讲干了什么，再找个人审一下。

### Pull Request 合并前检查清单

在合并前，请确认：

```text
[ ] 当前 PR 不是直接修改 main 后强行 push
[ ] 代码在自己的 feature 分支上完成
[ ] 没有提交 data/、.venv/、__pycache__/ 等无关文件
[ ] commit message 基本清楚
[ ] 至少运行过一次相关测试
[ ] 其他组员已经看过主要修改
```

## 5. 不要提交这些文件

以下文件不要提交到 GitHub：

```text
__pycache__/
*.pyc
.venv/
venv/
.env
data/
datasets/
checkpoints/
logs/
*.pkl
*.npy
*.npz
.DS_Store
.vscode/
```

原因：

- 数据集文件太大
- 模型 checkpoint 太大
- 缓存文件没有意义
- 每个人本地环境不同

---

## 6. 推荐 .gitignore

（我已经在项目根目录创建好了）

项目根目录建议创建 `.gitignore` 文件，内容如下：

```gitignore
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
.venv/
venv/
env/

# Environment variables
.env

# Data files
data/
datasets/

# Model checkpoints
checkpoints/
*.pkl
*.npy
*.npz

# Logs and outputs
logs/
runs/
outputs/

# Jupyter
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# Editor files
.vscode/
.idea/
```

如果确实需要提交某个小数据文件，微信群里吱一声。

---

## 7. 常见问题

1. 确认是否影响整个项目（区别在是否push了）。
2. 如果是，微信群里吱一声；不是，问AI怎么解决。

---

## 8. 推荐 commit message 格式（如果有余力的话）

推荐使用下面格式：

```text
动作: 简短说明
```

常用动作：

```text
Add: 新增功能
Fix: 修复问题
Update: 更新已有功能
Refactor: 重构代码
Docs: 修改文档
Experiment: 添加实验
```

示例：

```bash
git commit -m "Add: MNIST data loader"
git commit -m "Fix: softmax overflow issue"
git commit -m "Update: training loop with validation accuracy"
git commit -m "Refactor: split layers and activations"
git commit -m "Docs: add Git workflow guide"
git commit -m "Experiment: compare learning rates"
```

---

## 9. 最重要的三条规则

只记住三件事也可以：

1. **不要直接在 main 分支写代码**
2. **每次开始写代码前先 pull 最新代码**
3. **每次 push 前先确认代码能运行**
