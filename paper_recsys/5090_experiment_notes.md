# 5090 实验须知

## SSH 连接方式

```bash
ssh charlie@www.ultrapp.online
```

- 免密登录（已配置密钥）
- 主机名：charlie-ultra
- 用户名：charlie
- 项目路径：~/project/RecBole
- Conda 环境激活：`source scripts/recbole_env.sh`
- GPU：NVIDIA RTX 5090

## 注意事项

- **不要用** `a58@a58-Super-Server` 或 `198.18.0.61`（内网地址，从外部无法连接）
- sshd 监听端口：22 和 53729
- 磁盘空间：/dev/nvme0n1p10 约 1.9T（截至 2026-08-16 使用 84%）
