# Model handler
from julian.with_tf import Julian, init
from julian.handler.model_handler import ModelHandler, MODE
import os, glob


class NaicsHandler(ModelHandler):

    def __init__(self, mode=MODE.STREAM):
        super(NaicsHandler, self).__init__()
        args = init()
        args.mode = "predict"
        args.data_path = None
        args.model_dir = "models/naics"
        args.vocab_path = "data/naics/vocab.pickle"
        args.d3_table_path = "data/naics/l1_table.pickle"
        args.naics_code_path = "data/naics/l2_table.pickle"
        args.restore = True
        args.pred_output_path = None
        args.max_doc = 50
        args.batch_size = 1024
        args.dropout = 1.0
        args.name = "naics_stream"
        if mode == MODE.STREAM:
            self.init()
            args.input_stream = self.in_queue.receive_messages()
        
        self.fetch_all()
        self.julian = Julian(args) 

    def fetch_all(self):
        # FIXME
        remote_paths = (
            'julian/modles/naics/cnn-358000.data-00000-of-00001',
            'julian/models/naics/cnn-358000.meta',
            'julian/models/naics/cnn-358000.index',
        )
        list(map(lambda p:self.fetch_from_s3(\
                p, os.path.join(args.model_dir, \
                os.path.basename(p))), remote_paths))

        remove_paths = (
            'julian/data/naics/codes_3digits.csv',
            'julian/data/naics/vocab.pickle',
        )
        list(map(lambda p:self.fetch_from_s3(\
                p, os.path.join(os.path.dirname(args.vocab_path), \
                os.path.basename(p))), remote_paths))

    def run(self):
        # TODO
        for res in self.julian.run():
            self.out_queue.send_message(res.todict())


if __name__ == '__main__':
    NaicsHandler()()
