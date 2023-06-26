
import os

# 用来处理图片和标签数量不匹配的问题
XmlFilePath = 'Annotations/'
xmlFileName = os.listdir(XmlFilePath)
xmlNum = len(xmlFileName)

JpgFilePath = 'labels/'
jpgFileName = os.listdir(JpgFilePath)
jpgNum = len(jpgFileName)

xmlList = []
jpgList = []
deleteList = []

if xmlNum > jpgNum:            # 标签数量更多
    for i in jpgFileName:
        i = i.split('.')[0]
        jpgList.append(i)

    for i in xmlFileName:
        j = i.split('.')[0]
        if j not in jpgList:
            deleteList.append(i)

    print("多出来的标签为：")
    for i in deleteList:
        print(i)

elif xmlNum < jpgNum:          # 图片数量更多
    for i in xmlFileName:
        i = i.split('.')[0]
        xmlList.append(i)

    for i in jpgFileName:
        j = i.split('.')[0]
        if j not in xmlList:
            deleteList.append(i)

    print("多出来的图片为：")
    for i in deleteList:
        print(i)
else:
    print("图片与标签数量相等。")

# with open("ImageSets/Main/test.txt", "r") as file:
#     lines = file.readlines()
#
# lines = [line for line in lines if line.strip() not in deleteList]
#
# with open("ImageSets/Main/test.txt", "w") as file:
#     file.writelines(lines)

# 定义文件夹路径
# folder_path = 'labels'
# # 遍历文件夹
# for filename in os.listdir(folder_path):
#     # 获取文件的完整路径
#     file_path = os.path.join(folder_path, filename)
#
#     # 检查文件名是否为 '1'，如果是则删除文件
#     if filename in deleteList:
#         os.remove(file_path)
