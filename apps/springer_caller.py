# Dummy caller to Julian
from julian.with_tf import Julian, init
import pandas as pd
#import yaml


def chunk_stream():
    yield from [
        [
        "Recording with both parallel and orthogonal linearly polarized lights, polarization",
        "The classical barotropic model is used to indicate the possible connection between the intensification of atmos",
        "The age groups under study were: autospores (at the beginning of the light period), growing cells"
        ], [
        "Die Lebhaftigkeit der Tänze, gemessen durch die Tanzdauer, läßt sich durch Zusatz ätherischer Öle",
        "The X-ray crystal structure of the title compound, C 10 H 14 S 1 , is reported. Crystals are monoclinic",
        ]
    ]*100

def get_reader():
    data_path = "./data/springer/full.csv"
    reader = pd.read_csv(data_path, header=0, delimiter="#", chunksize=1024)
    for chunk in reader:
        yield chunk['desc']

def run_springer():

    # reader = chunk_stream()
    reader = get_reader()

    args = init()
    args.mode = "predict"
    args.data_path = None
    args.model_dir = "/mnt/data/zex/citronen_latest/models/springer/50_1024_07_vocab/"
    args.vocab_path = "/mnt/data/zex/citronen_latest/data/springer/lang/vocab.pickle"
    args.l1_table_path = "/mnt/data/zex/citronen_latest/data/springer/l1_table.pickle"
    args.l2_table_path = "/mnt/data/zex/citronen_latest/data/springer/l2_table.pickle"
    args.restore = True
    args.pred_output_path = None
    args.max_doc = 50
    args.batch_size = 1024
    args.dropout = 1.0
    args.name = "springer_stream"
    args.input_stream = reader

    julian = Julian(args)
    for res in julian.run():
        # iid, l1, l2
        print(res['iid'], res['l1'], res['l2'])
        print(res.iloc[0,:])


if __name__ == '__main__':
    run_springer()
