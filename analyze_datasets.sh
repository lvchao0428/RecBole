#!/bin/bash
# 数据集统计分析便捷脚本

# 使用说明
usage() {
    echo "数据集统计分析工具"
    echo ""
    echo "用法:"
    echo "  $0 example              # 分析示例数据集"
    echo "  $0 full <server_path>   # 分析服务器完整数据集"
    echo "  $0 help                 # 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 example"
    echo "  $0 full /data/recbole/datasets"
    exit 1
}

# 检查参数
if [ $# -eq 0 ]; then
    usage
fi

case "$1" in
    example)
        echo "分析示例数据集 (ml-1m-example 和 Amazon_Beauty-example)..."
        python3 dataset_statistics.py --example
        ;;
    
    full)
        if [ -z "$2" ]; then
            echo "错误: 请提供服务器数据集路径"
            echo "用法: $0 full <server_path>"
            exit 1
        fi
        echo "分析完整数据集 (ml-1m 和 Amazon_Beauty)..."
        echo "服务器路径: $2"
        python3 dataset_statistics.py --full --server-path "$2"
        ;;
    
    help|--help|-h)
        usage
        ;;
    
    *)
        echo "错误: 未知参数 '$1'"
        usage
        ;;
esac


