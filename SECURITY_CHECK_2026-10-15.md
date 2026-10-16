# SECURITY_CHECK_2026-10-15 — Panthera-HT

静态扫描(正则,非完整审计)。严重度 H/M/L。真实密钥值未写入本报告,仅列位置。

## 中危 (1)

| 文件:行 | 类别 | 问题 | 建议修法 |
|---|---|---|---|
| `Panthera-HT_SDK/panthera_cpp/motor_cpp/third_part/lcm/format_code.sh:142` | code | shell eval of variable | 避免 shell=True/不安全反序列化,改用参数列表、safe_load、json 等 |

## 低危 (3)

| 文件:行 | 类别 | 问题 | 建议修法 |
|---|---|---|---|
| `Panthera-HT_Host/Panthera_digital_twin-main/backend/app.py:1544` | code | eval/exec | 避免 shell=True/不安全反序列化,改用参数列表、safe_load、json 等 |
| `Panthera-HT_Host/Panthera_digital_twin-main/backend/run_script.py:82` | code | eval/exec | 避免 shell=True/不安全反序列化,改用参数列表、safe_load、json 等 |
| `Panthera-HT_SDK/panthera_cpp/motor_cpp/third_part/lcm/docker/Dockerfile.fedora:0` | docker | no non-root USER (runs as root) | 添加 USER 非 root |

## Dependabot 未关闭告警

| 严重度 | 包 | GHSA | manifest |
|---|---|---|---|
| high | nanoid | GHSA-xwg4-73v4-xw9w | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| high | socket.io-parser | GHSA-2m8v-j782-fhvr | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| medium | postcss | GHSA-fxqj-rqcc-2cmp | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| high | postcss | GHSA-r28c-9q8g-f849 | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| high | postcss | GHSA-6g55-p6wh-862q | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| medium | vite | GHSA-v6wh-96g9-6wx3 | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| high | vite | GHSA-fx2h-pf6j-xcff | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| high | ws | GHSA-96hv-2xvq-fx4p | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| medium | ws | GHSA-58qx-3vcg-4xpx | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| medium | postcss | GHSA-qx2v-qp2m-jg93 | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| medium | vite | GHSA-4w7w-66w2-5vf9 | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| high | socket.io-parser | GHSA-677m-j7p3-52f9 | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| high | rollup | GHSA-mw96-cpmx-2vgc | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |
| medium | esbuild | GHSA-67mh-4wv8-2f99 | Panthera-HT_Host/Panthera_digital_twin-main/frontend/package-lock.json |

## 通用建议
- pull_request_target:不要 checkout PR head 代码并执行;或改用 pull_request。
- 工作流中不要把 `${{ github.event.* }}` 直接插入 run:,改用 env 变量。
- 第三方 action 固定到 commit SHA;工作流加 `permissions: contents: read`。

本 PR 仅添加此报告,未改动代码。