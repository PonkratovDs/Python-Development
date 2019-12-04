import sys

def main(args):
    m_br, m_mes = args[1], args[2]
    fd = open("log.txt", 'r')
    for line in fd:
        if line.find('[' + m_br + ']') != -1 and line.find(m_mes) != -1:
            print(line.rstrip())
    fd.close()

if __name__ == '__main__':
    main(sys.argv)