from pathlib import Path


def read_drs():
    result = []
    file_path = Path('../pmb-3.0.0-sample/exp_data/en/gold/train.txt')
    drs = DRS()
    with file_path.open() as f:
        for line in f:
            if line.startswith(r'%%%') or line.strip() == '':
                continue
            elif line.startswith('% '):
                result.append(drs)
                drs = DRS()
            else:

                # Parse line
                elem = line.split('%')[0].strip().split(' ')
                print(elem)

                # Handle box
                i = int(elem[0].replace('b', ''))
                if i not in drs.box:
                    drs.add_box(Box(i))

                # DRS operator
                if len(elem[1]) == 3 and elem[1].isupper():

                    if elem[1] == 'REF':
                        drs.box[i].add_ref(elem[2])
                    elif elem[1] in {'NOT', 'IMP'}:
                        assert elem[2].startswith('b')
                        j = int(elem[2].replace('b', ''))
                        drs.box[i].add_relation(j, elem[1])

                # Semantic role
                elif elem[1][0].isupper() and elem[1][1:].islower():
                    assert len(elem) == 4
                    drs.box[i].add_sem_role(elem[1], elem[2], elem[3])

                # constant
                elif elem[1].islower():
                    assert elem[2].startswith('"') and elem[2].endswith('"')
                    assert len(elem) == 4
                    concept = elem[2].strip('"')
                    arg = elem[3]
                    drs.box[i].add_cond(elem[1], concept, arg)
                elif elem[1].isupper() and len(elem[1]) > 3:
                    continue
                else:
                    raise ValueError('Unhandled case in column 1')

    return result


class Box:
    def __init__(self, i):
        self.id = i
        self.ref = []
        self.cond = []
        self.relation = []
        self.prsp = []
        return

    def add_ref(self, x):
        self.ref.append(x)
        return None

    def add_cond(self, symbol, concept, *arg):
        self.cond.append((symbol, concept, *arg))
        return None

    def add_relation(self, i, name):
        self.relation.append((i, name))
        return None

    def add_sem_role(self, role, *arg):
        self.cond.append((role, *arg))
        return None

    def add_prsp(self):
        return None


class DRS:
    def __init__(self):
        self.raw = ''
        self.box = dict()
        return

    def add_box(self, box):
        assert box.id not in self.box
        self.box[box.id] = box
        return None

    def to_fol(self):
        return


class Tag:
    def __init__(self):
        self.id = ''

        self.tok = ''
        self.lemma = ''
        self.sem = ''  # semantic tag (Bjerva etal 2016)
        self.sym = ''  # non-logical constant (lemmatization, normalization, lexically disambigoution)
        self.sense = ''
        self.verbnet = ''
        self.wordnet = ''
        self.from_ = ''
        self.to_ = ''
        return
