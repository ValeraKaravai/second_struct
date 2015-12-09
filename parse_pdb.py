import argparse


def parse_dssp(input_file):
    amino_chain = {}
    second_struc_chain = {}
    for i in open(input_file):
        i = i.split()
        if i[0] == 'ASG':
            amino = i[1][0]
            second = i[5]
            chain = i[2]
            if chain in amino_chain:
                amino_chain[chain].append(amino)
                second_struc_chain[chain].append(second)
            else:
                amino_chain.update({chain: [amino]})
                second_struc_chain.update({chain: [second]})
    return amino_chain, second_struc_chain


def write(amino_chain, second_chain, input_file):
    input_file = input_file.split('.')
    file_list = []
    str_chain_amino = []
    str_chain_struc = []
    for i in amino_chain:
        str_file = 'parse_file/' + str(input_file[0]) + '.' + str(i) + '.txt'
        file_list.append(str_file)
    for i in amino_chain:
        str_c = ''
        for j in amino_chain[i]:
            str_c += str(j)
        str_chain_amino.append(str_c)
    for i in second_chain:
        str_c = ''
        for j in second_chain[i]:
            str_c += str(j)
        str_chain_struc.append(str_c)
    for i in range(len(str_chain_struc)):
        out = open(file_list[i], 'w+')
        out.write(str(str_chain_amino[i]) + '\n' + str(str_chain_struc[i]))


def arg_parse():
    pars = argparse.ArgumentParser()
    pars.add_argument("-i", "--input", help="input file dssp")
    args = pars.parse_args()
    return args


def main():
    a = arg_parse()
    input_file = a.input
    amino_chain, second_chain = parse_dssp(input_file)
    write(amino_chain, second_chain, input_file)
if __name__ == "__main__":
    main()
