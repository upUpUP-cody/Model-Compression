# VS Code 远程服务器使用 Codex

> 来自 Codex

## 使用步骤

### 1. 启动本机代理

打开西柚并启用代理模式，然后启动 gost：

```powershell
gost.exe -L "http://127.0.0.1:18080"
```

### 2. 建立 SSH 反向隧道

在 Windows 执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\windows-setup-tunnel.ps1
```

保持窗口运行。隧道断开后会自动重连。

### 3. 启动服务器代理

服务器重启后执行：

```bash
bash /root/codex-xiyou-setup/start_proxy_stack.sh
```

验证代理：

```bash
curl -x http://127.0.0.1:17890 https://api.openai.com/v1/models
```

返回 `401` 通常表示网络已连通。

### 4. 连接 VS Code

1. 安装 `Remote - SSH` 扩展。
2. 连接远程服务器。
3. 在远程窗口安装 `OpenAI Codex / ChatGPT` 扩展。
4. 执行 `Developer: Reload Window`。
5. 打开 Codex 面板。

## 启动顺序

```text
西柚代理 → gost:18080 → SSH 反向隧道 → 服务器代理 → VS Code Remote-SSH → Codex
```

## 常见问题

- `ECONNREFUSED 127.0.0.1:15721`：旧代理端口，应使用 `17890`。
- `Codex could not start`：确认隧道和服务器代理已启动，然后重新加载 VS Code。
- 代理无响应：确认 gost 使用 `18080`，不要把 PAC 端口 `7525` 作为转发目标。
