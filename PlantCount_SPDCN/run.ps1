# 获取当前时间戳，格式为 mmddHHMMSS
$T = Get-Date -Format "MMddHHmmss"

# 创建目录结构
New-Item -ItemType Directory -Path "exp/$T/code" | Out-Null
New-Item -ItemType Directory -Path "exp/$T/train.log" | Out-Null

# 复制代码文件和资源
Copy-Item -Path "datasets" -Destination "exp/$T/code/datasets" -Recurse
Copy-Item -Path "models" -Destination "exp/$T/code/models" -Recurse
Copy-Item -Path "*.py" -Destination "exp/$T/code/"
Copy-Item -Path "run.sh" -Destination "exp/$T/code/"

# 设置数据集路径（注意使用双引号包裹路径）
$datapath = "D:\Code\Workspace\SPDCN-CAC\datasets\FSC147_384_V2"

# 激活 conda 环境（可选，确保你已激活环境或在脚本前手动激活）
conda activate SPDCN

# 执行 Python 命令并记录日志
Start-Process python -ArgumentList "main.py --data-path $datapath --batch-size 16 --accumulation-steps 1 --tag $T" -RedirectStandardOutput "exp/$T/train.log/running.log" -Wait -NoNewWindow