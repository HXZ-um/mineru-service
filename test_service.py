#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本，用于验证MinerU PDF解析服务是否正常工作
"""

import requests
import json
import time

def test_health_check():
    """测试服务是否正常运行"""
    try:
        response = requests.get("http://localhost:5002/docs")
        if response.status_code == 200:
            print("✓ 服务健康检查通过")
            return True
        else:
            print(f"✗ 服务健康检查失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ 无法连接到服务: {e}")
        return False

def main():
    print("MinerU PDF解析服务测试脚本")
    print("=" * 40)
    
    # 测试服务是否运行
    if not test_health_check():
        print("请确保服务已在端口5002上运行")
        return
    
    print("\n服务测试完成!")

if __name__ == "__main__":
    main()