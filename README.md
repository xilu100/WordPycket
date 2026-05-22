# WordPycket

一个使用 DDD 洋葱架构组织的 Python Tkinter GUI 小软件，用来从 CSV 导入词库，并按掌握程度进行单词复习。

数据层使用 Python 标准库内置的 SQLite，不需要额外安装数据库服务。SQLite 支持事务、索引、约束、视图、触发器、CTE、JSON 扩展等常见 SQL 能力，但它不是 PostgreSQL 的完整同等替代品；如果后续需要多用户并发、网络访问、复杂权限和 PostgreSQL 扩展，可以在 `infrastructure` 层替换仓储实现。

## 当前功能

- 启动时读取 `input/word_frequency.csv`。
- 将 CSV 中的 `Index`、`English`、`Chinese`、`Frequency`、`Forms` 存入 SQLite。
- 重复启动会按单词更新释义、频率和词形，不会清空已有复习进度。
- 复习界面同时显示英文单词和中文释义。
- 可用“上一个”“下一个”按 CSV 顺序浏览词卡。
- 点击“会”会提高掌握等级，后续更少出现。
- 点击“不会”会降低掌握等级，后续更容易反复出现。
- 右侧列表可搜索词库，并查看频率、状态和会/不会次数。

## 运行

```powershell
python -m wordpycket.main
```

如果没有安装为包，请在项目根目录使用：

```powershell
$env:PYTHONPATH="src"; python -m wordpycket.main
```

## 架构

- `domain`：领域实体和仓储接口，不依赖外层实现。
- `application`：应用服务，编排用例。
- `infrastructure`：SQLite 仓储实现和 CSV 导入器。
- `presentation`：Tkinter GUI。

依赖方向为：`presentation -> application -> domain`，`infrastructure -> domain`。
