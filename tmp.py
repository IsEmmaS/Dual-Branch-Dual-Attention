import sys
import ast
import astor
import os

def remove_comments(source_code):
    """
    移除Python源代码中的所有注释。
    """
    try:
        # 将源代码解析为AST
        tree = ast.parse(source_code)
        
        # 将AST转换回源代码，不包含注释
        new_code = astor.to_source(tree)
        
        return new_code
    except SyntaxError as e:
        print(f"语法错误: {e}")
        return None

def main(file_path):
    if not os.path.isfile(file_path):
        print(f"文件不存在: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()

    cleaned_code = remove_comments(source_code)

    if cleaned_code is not None:
        # 生成输出文件路径
        output_file = file_path
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_code)
        
        print(f"已移除注释，保存到 {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python remove_comments.py <python_file_path>")
    else:
        main(sys.argv[1])