import massage, sys

def main():
    datafileA = open('../data/2016downloaded4-subtask A.tsv')
    taskA = massage.massage(datafileA, 2)
    print(taskA)

if __name__ == '__main__':
    sys.exit(main())