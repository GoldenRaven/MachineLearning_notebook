csv_file = open('../test-me/test.csv', encoding='gbk')
line = next(csv_file)
# print(line)
for line in csv_file:
    pass
print(line)
csv_file.close()

with open('../test-me/test.csv', encoding='gbk') as csv_file: # 只在文件打开时运行，并自动close
    for line in csv_file:
        pass
print(line)
