# Model handler
import os, glob
from julian.core.with_tf import init
from julian.handler.model_handler import ModelHandler, MODE


class SpringerHandler(ModelHandler):

    def __init__(self, mode=MODE.STREAM, **kwargs):
        super(SpringerHandler, self).__init__(**kwargs)
        args = init()
        args.mode = "predict"
        args.data_path = None
        args.model_dir = os.path.join(os.getcwd(), "models/springer/")
        args.vocab_path = "data/springer/vocab.pickle"
        args.l1_table_path = "data/springer/l1_table.pickle"
        args.l2_table_path = "data/springer/l2_table.pickle"
        args.restore = True
        args.pred_output_path = None
        args.max_doc = 50
        args.batch_size = 1024
        args.dropout = 1.0
        args.name = "springer_stream"
        args.input_stream = []
        self.fetch_all(args)
        self.setup_model(args)

    def fetch_all(self, args):
        # FIXME
        remote_base = 'config/julian/models/springer'
        remote_paths = (
                os.path.join(remote_base, 'cnn-119000.data-00000-of-00001'),
                os.path.join(remote_base, 'cnn-119000.meta'),
                os.path.join(remote_base, 'cnn-119000.index'),
                os.path.join(remote_base, 'checkpoint'),
        )
        list(map(lambda p:self.fetch_from_s3(\
                p, os.path.join(args.model_dir, \
                os.path.basename(p)), force=False), remote_paths))

        remote_base = 'config/julian/data/springer'
        remote_paths = (
                os.path.join(remote_base, 'l1_table.pickle'),
                os.path.join(remote_base, 'l2_table.pickle'),
                os.path.join(remote_base, 'lang/vocab.pickle'),
                )
        list(map(lambda p:self.fetch_from_s3(\
                p, os.path.join(os.path.dirname(args.vocab_path), \
                os.path.basename(p)), force=False), remote_paths))


if __name__ == '__main__':
    hdr = SpringerHandler()
    for x in hdr.run_async():
        print(x)
