import pandas as pd
import matplotlib.pyplot as plt
import random
import json


def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = [float(value) for line in file.readlines() for value in line.strip().split()]
    print(data)
    return data


def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(fp=f)
    return data


def generate_data(data, tim):
    last_tim = tim[0]
    tmp = 0
    cnt = 0
    res = []
    for i, x in enumerate(data):
        if tim[i] - last_tim < 1:
            tmp += data[i]
            cnt += 1
        else:
            last_tim += 1
            if cnt == 0:
                res.append(0)
            else:
                res.append(tmp / cnt)
            cnt = 1
            tmp = data[i]
    return res


def draw(dataset, rate):
    vllmDataPath = f'logs/[{dataset}][10k][0][0][0][{rate}][vllm].json'
    mlfqDataPath = f'logs/[{dataset}][10k][2][8][5.0][{rate}][None].json'
    vllmData = read_data(vllmDataPath)
    print(vllmData.keys())
    mlfqData = read_data(mlfqDataPath)
    fig, axs = plt.subplots(4, 1, figsize=(20, 16))

    mlfqUTL = generate_data(mlfqData["utl"], mlfqData["utl_time"])
    axs[0].plot([x for x in range(len(mlfqUTL))], mlfqUTL, label='MLFQ', color="blue")
    vllmUTL = generate_data(vllmData["utl"], vllmData["utl_time"])
    axs[0].plot([x for x in range(len(vllmUTL))], vllmUTL, label='vllm', color="red")
    axs[0].legend(fontsize=22)
    axs[0].set_title(f"utl [{dataset}][{rate}]", fontsize=24, fontweight='bold')
    axs[0].set_xlabel("Time/s", fontsize=24)
    axs[0].set_ylabel("utilization", fontsize=24)

    mlfqRe = generate_data(mlfqData["recomputed_times"], mlfqData["utl_time"])
    axs[1].plot([x for x in range(len(mlfqRe))], mlfqRe, label='MLFQ', color="blue")
    vllmRe = generate_data(vllmData["recomputed_times"], vllmData["utl_time"])
    axs[1].plot([x for x in range(len(vllmRe))], vllmRe, label='vllm', color="red")
    axs[1].legend(fontsize=22)
    axs[1].set_title(f"recomputed_times per second [{dataset}][{rate}]", fontsize=24, fontweight='bold')
    axs[1].set_xlabel("Time/s", fontsize=24)
    axs[1].set_ylabel("Count", fontsize=24)

    mlfqJCT = []
    vllmJCT = []
    for i in range(len(mlfqData["JCT"]) - 1, 0, -1):
        mlfqData["JCT"][i] = mlfqData["JCT"][i] * (i + 1) - mlfqData["JCT"][i - 1] * i
    for i in range(len(vllmData["JCT"]) - 1, 0, -1):
        vllmData["JCT"][i] = vllmData["JCT"][i] * (i + 1) - vllmData["JCT"][i - 1] * i
    mlfqData["JCT_times"][0] = 0
    vllmData["JCT_times"][0] = 0
    tmp = 0
    cnt = 0
    last_tim = 0
    for i, tim in enumerate(mlfqData["JCT_times"]):
        if tim - last_tim < 1:
            tmp += mlfqData["JCT"][i]
            cnt += 1
        else:
            if cnt == 0:
                mlfqJCT.append(0)
            else:
                mlfqJCT.append(tmp / cnt)
            tmp = mlfqData["JCT"][i]
            cnt = 1
            last_tim += 1
    tmp = 0
    cnt = 0
    last_tim = 0
    for i, tim in enumerate(vllmData["JCT_times"]):
        if tim - last_tim < 1:
            tmp += vllmData["JCT"][i]
            cnt += 1
        else:
            if cnt == 0:
                vllmJCT.append(0)
            else:
                vllmJCT.append(tmp / cnt)
            tmp = vllmData["JCT"][i]
            cnt = 1
            last_tim += 1
    axs[2].plot([x for x in range(len(mlfqJCT))], mlfqJCT, label='MLFQ', color="blue")
    axs[2].plot([x for x in range(len(vllmJCT))], vllmJCT, label='vllm', color="red")
    axs[2].legend(fontsize=22)
    axs[2].set_title(f"JCT [{dataset}][{rate}]", fontsize=24, fontweight='bold')
    axs[2].set_xlabel("Time/s", fontsize=24)
    axs[2].set_ylabel("JCT/s", fontsize=24)

    mlfqData["e2e_times"].sort()
    vllmData["e2e_times"].sort()
    t = [(x + 1) / len(mlfqData["e2e_times"]) for x in range(len(mlfqData["e2e_times"]))]
    axs[3].plot(t, mlfqData["e2e_times"], label='MLFQ', color="blue")
    axs[3].plot(t, vllmData["e2e_times"], label='vllm', color="red")
    axs[3].legend(fontsize=22)
    axs[3].set_title(f"Percentage Distribution of Assignment Completion Times [{dataset}][{rate}]",
                     fontsize=24, fontweight='bold')
    axs[3].set_xlabel("%", fontsize=24)
    axs[3].set_ylabel("Time/s", fontsize=24)
    axs[3].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])

    for ax in axs:
        ax.legend(fontsize=22, ncol=2, loc='upper left')
        ax.tick_params(axis='both', labelsize=20)

    plt.subplots_adjust(hspace=10)
    plt.tight_layout()

    plt.savefig(f"fig/[{dataset}][{rate}].pdf")
    plt.savefig(f"fig/[{dataset}][{rate}].png")


datasets = ["alpaca",
            "dolly",
            "alpaca_python",
            "mmlu"
            ]
rates = [0.01,
         0.02,
         0.05
         ]
for dataset in datasets:
    for rate in rates:
        draw(dataset, rate)
        # plt.show()
        # exit(0)

# 创建一个包含多个子图的布局
# fig, axs = plt.subplots(6, 1, figsize=(16,36))
#
# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-235.txt'
# file2_path = 'FastServe-235.txt'
# file3_path = 'FastServe-SC-235.txt'
# file4_path = 'FastServe-SCopt-235.txt'
#
# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)
# data4 = read_data_from_file(file4_path)
#
# # plt.plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# # plt.show()
#
# # 绘制第一个子图
# axs[0].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[0].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[0].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[0].plot(data4[0::2], data4[1::2], label='FastServe-SCopt', color='purple')
# axs[0].set_title('235')
# axs[0].set_xlabel('Time(s)')
# axs[0].set_ylabel('JCT(s)')
# axs[0].legend(loc='upper left')
#
#
# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-253.txt'
# file2_path = 'FastServe-253.txt'
# file3_path = 'FastServe-SC-253.txt'
# file4_path = 'FastServe-SCopt-253.txt'
#
# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)
# data4 = read_data_from_file(file4_path)
#
# # 绘制第二个子图
# axs[1].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[1].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[1].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[1].plot(data4[0::2], data4[1::2], label='FastServe-SCopt', color='purple')
# axs[1].set_title('253')
# axs[1].set_xlabel('Time(s)')
# axs[1].set_ylabel('JCT(s)')
# axs[1].legend(loc='upper left')
#
#
# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-325.txt'
# file2_path = 'FastServe-325.txt'
# file3_path = 'FastServe-SC-325.txt'
# file4_path = 'FastServe-SCopt-325.txt'
#
# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)
# data4 = read_data_from_file(file4_path)
#
# # 绘制第三个子图
# axs[2].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[2].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[2].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[2].plot(data4[0::2], data4[1::2], label='FastServe-SCopt', color='purple')
# axs[2].set_title('325')
# axs[2].set_xlabel('Time(s)')
# axs[2].set_ylabel('JCT(s)')
# axs[2].legend(loc='upper left')
#
#
# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-352.txt'
# file2_path = 'FastServe-352.txt'
# file3_path = 'FastServe-SC-352.txt'
# file4_path = 'FastServe-SCopt-352.txt'
#
# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)
# data4 = read_data_from_file(file4_path)
#
# # 绘制第四个子图
# axs[3].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[3].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[3].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[3].plot(data4[0::2], data4[1::2], label='FastServe-SCopt', color='purple')
# axs[3].set_title('352')
# axs[3].set_xlabel('Time(s)')
# axs[3].set_ylabel('JCT(s)')
# axs[3].legend(loc='upper left')
#
#
# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-523.txt'
# file2_path = 'FastServe-523.txt'
# file3_path = 'FastServe-SC-523.txt'
# file4_path = 'FastServe-SCopt-523.txt'
#
# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)
# data4 = read_data_from_file(file4_path)
#
# # 绘制第五个子图
# axs[4].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[4].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[4].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[4].plot(data4[0::2], data4[1::2], label='FastServe-SCopt', color='purple')
# axs[4].set_title('523')
# axs[4].set_xlabel('Time(s)')
# axs[4].set_ylabel('JCT(s)')
# axs[4].legend(loc='upper left')
#
#
# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-532.txt'
# file2_path = 'FastServe-532.txt'
# file3_path = 'FastServe-SC-532.txt'
# file4_path = 'FastServe-SCopt-532.txt'
#
# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)
# data4 = read_data_from_file(file4_path)
#
# # 绘制第六个子图
# axs[5].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[5].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[5].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[5].plot(data4[0::2], data4[1::2], label='FastServe-SCopt', color='purple')
# axs[5].set_title('532')
# axs[5].set_xlabel('Time(s)')
# axs[5].set_ylabel('JCT(s)')
# axs[5].legend(loc='upper left')

# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-allSmall.txt'
# file2_path = 'FastServe-allSmall.txt'
# file3_path = 'FastServe-SC-allSmall.txt'

# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)

# # 绘制第七个子图
# axs[2, 0].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[2, 0].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[2, 0].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[2, 0].set_title('all Short')
# axs[2, 0].set_xlabel('Finish job')
# axs[2, 0].set_ylabel('JCT(s)')
# axs[2, 0].legend(loc='lower right')
#
# # 文件路径需要根据实际情况修改
# file1_path = 'FCFS-allBig.txt'
# file2_path = 'FastServe-allBig.txt'
# file3_path = 'FastServe-SC-allBig.txt'
#
# data1 = read_data_from_file(file1_path)
# data2 = read_data_from_file(file2_path)
# data3 = read_data_from_file(file3_path)
#
# # 绘制第八个子图
# axs[2, 1].plot(data1[0::2], data1[1::2], label='FCFS', color='blue')
# axs[2, 1].plot(data2[0::2], data2[1::2], label='FastServe', color='red')
# axs[2, 1].plot(data3[0::2], data3[1::2], label='FastServe-SC', color='green')
# axs[2, 1].set_title('all Long')
# axs[2, 1].set_xlabel('Finish job')
# axs[2, 1].set_ylabel('JCT(s)')
# axs[2, 1].legend(loc='lower right')

# plt.subplots_adjust(wspace=0.25, hspace=0.25)
#
# plt.show()
