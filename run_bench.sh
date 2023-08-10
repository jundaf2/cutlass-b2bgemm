make clean && make 13_fused_two_gemms_f16_sm80_rf

program="./13_fused_two_gemms_f16_sm80_rf"
param_file="./params.txt"

# 使用while循环从文件中读取参数
while IFS= read -r param
do
    # 执行程序并传递参数
    $program $param
done < "$param_file"
