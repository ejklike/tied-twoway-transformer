from rdkit import Chem
from tqdm import tqdm
import pandas as pd


def disable_rdkit_log():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def canonicalize_smiles(smi):
    # assumes the input smiles contains no blank space chars.
    assert ' ' not in smi
    disable_rdkit_log()
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return ''


def canonicalize_smitxt(fin, fout, remove_score=False):
    with open(fin, 'r') as f:
        lines = f.readlines()

    def process_line(line):
        line = remove_space(line)
        if remove_score:
            smiles = line.split(',')[0]
        smiles_can = canonicalize_smiles(smiles)
        return smi_tokenizer(smiles_can) + '\n'

    lines = [process_line(line) for line in lines]
    
    with open(fout, 'w') as f:
        f.writelines(lines)


def linecount(f):
    try:
        count = sum(1 for line in open(f, 'r'))
    except FileNotFoundError:
        # print('file not found')
        count = -1
    return count


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    (https://github.com/pschwllr/MolecularTransformer)
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def tile_lines_n_times(fin, fout, n=10):
    with open(fin, 'r') as f:
        lines = f.readlines()
    with open(fout, 'w') as f:
        for line in lines:
            f.writelines([line] * n)



def remove_space(line):
    """
    remove space from tokenized line
    """
    return ''.join(line.strip().split(' '))


def normalize_smiles(smi, canonicalize=False):
    if canonicalize:
        smi = canonicalize_smiles(smi)
    tokens = list(set(smi.split('.')))
    return '.'.join(sorted(tokens))


def match_smiles(sm1, sm2, canonicalize=False):
    """
    check whether two smiles are identical or not
    """
    if canonicalize:
        sm1 = canonicalize_smiles(sm1)
        sm2 = canonicalize_smiles(sm2)
    
    if '' in [sm1, sm2]:
        return False

    sm1_set = set(sm1.split('.'))
    sm2_set = set(sm2.split('.'))

    sm1 = '.'.join(sorted(list(sm1_set)))
    sm2 = '.'.join(sorted(list(sm2_set)))
    return sm1 == sm2



def read_beam_file(file_path, beam_size=1, parse_func=None):
    """
    read smiles file into a nested list of the size [n_data, beam]
    """
    with open(file_path, 'r+') as f:
        lines = f.readlines() # N * B
        assert len(lines) % beam_size == 0
        n_chunk = len(lines) // beam_size

    if parse_func is None:
        parse_func = remove_space
    lines = [parse_func(line) for line in lines]

    if beam_size == 1:
        return lines
    else:
        return [lines[i*beam_size:(i+1)*beam_size] for i in range(n_chunk)]


def combine_N_translations(fwd_input_format, output_path, num_experts, 
                           beam_size, n_best, bwd_input_format=None, 
                           bwd_target_path=None):
    """
    Reads the output smiles from each of the latent classes and combines them.
    Args:
        input_dir: The path to the input directory containing output files
        n_latent: The number of latent classes used for the model
        n_best: Number of smiles results per reaction
        output_path: If given, writes the combined smiles to this path
    """
    bwd_target = None
    if bwd_target_path is not None:
        with open(bwd_target_path, 'r') as f:
            bwd_target = [remove_space(x) for x in f.readlines()]

    def parse(line):
        smiles, score = remove_space(line).split(',')
        return smiles, float(score)
    
    # results_path is the prefix for the different latent file outputs
    fwd_expert_output_list = []
    if bwd_input_format: # get scores or smiles
        bwd_expert_output_list = []

    for expert_id in range(num_experts):
        fwd_input_path = fwd_input_format % expert_id
        fwd_smiles_list = read_beam_file(
            fwd_input_path, beam_size=beam_size, parse_func=parse)
        fwd_expert_output_list.append(fwd_smiles_list)
        if bwd_input_format:
            bwd_file_path = bwd_input_format % expert_id
            bwd_smiles_list = read_beam_file(
                bwd_file_path, beam_size=beam_size, parse_func=parse)
            bwd_expert_output_list.append(bwd_smiles_list)
            assert len(fwd_smiles_list) == len(bwd_smiles_list)
    n_data = len(fwd_expert_output_list[0])

    combined_list = []
    output_file = open(output_path, 'w+')
    output_file_detailed = open(output_path + '.detailed', 'w+')

    for data_idx in tqdm(range(n_data)):
        # SMILES and scores will be saved here
        r_dict = {}
        # input SMILES (for plausibility check)
        bwd_target_smiles = bwd_target[data_idx] if bwd_target else ''
        for expert_id in range(num_experts):
            fwd_output_list = fwd_expert_output_list[expert_id][data_idx]
            if bwd_input_format:
                bwd_output_list = bwd_expert_output_list[expert_id][data_idx]
            for beam_idx, (smiles, score) in enumerate(fwd_output_list):
                smiles = normalize_smiles(smiles)

                bwd_smiles = ''
                if bwd_input_format:
                    bwd_smiles, bwd_score = bwd_output_list[beam_idx]
                    score += bwd_score
                
                update_condition = (
                    smiles not in r_dict or   # Add the output to dictionary
                    score > r_dict[smiles][0] # Update with the best score
                )
                if update_condition:
                    r_dict[smiles] = (score, expert_id, bwd_smiles)

        sorted_output = sorted(r_dict.items(), 
                               key=lambda x: x[1],
                               reverse=True)
        top_smiles = []
        for beam_idx in range(n_best):
            if beam_idx < len(sorted_output):
                smiles, (score, expert_id, bwd_smiles) = sorted_output[beam_idx]
                top_smiles += [smiles]
            else:
                smiles, (score, expert_id, bwd_smiles) = '', (-1e5, -1, '')

            cycle_correct = (1 if bwd_target_smiles == bwd_smiles else 0)

            output_file.write('%s\n' % smiles)
            output_file_detailed.write(
                '%s,%.4f,%d,%s,%d\n' 
                % (smiles, score, expert_id, bwd_smiles, cycle_correct))
        combined_list.append(top_smiles)
    
    output_file.close()
    output_file_detailed.close()
    
    return combined_list


def read_smitxt(fpath, canonicalize=False):
    """
    read txt that each line is smiles
    """
    with open(fpath, 'r') as f:
        lines = f.readlines()
    lines = [remove_space(l) for l in lines]
    if canonicalize:
        disable_rdkit_log()
        lines = [canonicalize_smiles(l) for l in lines]
    return lines


def get_rank(row, base, max_rank):
    target_smi = row['target'] # already canonicalized
    for i in range(1, max_rank+1):
        output_smi = row['{}{}'.format(base, i)]
        if match_smiles(output_smi, target_smi):
            return i
    return 0


def evaluate(output_path, target_path, n_best=10,
             save_score_to=None, save_rawdata=False):
    targets = read_smitxt(target_path) # N
    outputs = read_smitxt(output_path) # N * B

    test_df = pd.DataFrame(targets)
    test_df.columns = ['target']
    test_df['target'] = test_df['target'].apply(lambda x: normalize_smiles(x))
    total = len(test_df)

    print(len(outputs), len(targets))
    for i in range(n_best):
        test_df['pred_{}'.format(i + 1)] = outputs[i::n_best]
        test_df['pred_can_{}'.format(i + 1)] = \
            test_df['pred_{}'.format(i + 1)].apply(
                lambda x: normalize_smiles(x, canonicalize=True))
    test_df['rank'] = test_df.apply(
        lambda row: get_rank(row, 'pred_can_', n_best), axis=1)

    correct = 0
    invalid_smiles = 0
    duplicate_smiles = 0
    smiles_sets = [set() for _ in range(total)]

    accuracy_list = []
    invalid_list = []
    duplicate_list = []

    for i in range(1, n_best+1):
        correct += (test_df['rank'] == i).sum()
        invalid_smiles += (test_df['pred_can_{}'.format(i)] == '').sum()

        for j, x in enumerate(test_df['pred_can_{}'.format(i)]):
            if x != '':
                if x in smiles_sets[j]:
                    duplicate_smiles += 1
                else:
                    smiles_sets[j].add(x)
            else:
                duplicate_smiles += 1

        accuracy = correct/total * 100
        invalid_ratio = invalid_smiles/(total*i) * 100
        numbers = [i, accuracy, invalid_ratio]
        print('Top-{}:\t{:.2f}% || Invalid {:.2f}%'.format(*numbers))
        with open(save_score_to, 'a+') as f:
            f.write(','.join(['%g' %x for x in numbers]) + '\n')

    unique_ratio = (1 - duplicate_smiles/(total*(n_best))) * 100
    print('Unique ratio:\t{:.2f}%'.format(unique_ratio))
    with open(save_score_to, 'a+') as f:
        f.write('%g\n' % unique_ratio)

    if save_rawdata:
        test_df.to_csv(save_score_to + '.rawdata', index=False)