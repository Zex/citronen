# Start all handlers
# Author: Zex Li <top_zlynch@yahoo.com>
import multiprocessing as mp
from julian.common.config import get_config
from julian.handler.springer_handler import SpringerHandler
from julian.handler.naics_handler import NaicsHandler

def run_async(hdr_name):
    hdr = globals().get(hdr_name)()
    for res in hdr.run_async():
        print(res)


def start():
    pool = []
    try:
        for hdr_name in get_config().model_handlers.split(','):
            pool.append(mp.Process(target=run_async,
                                    args=(hdr_name,),
                                    name=hdr_name))
        list(map(lambda p: p.start(), pool))
    except KeyboardInterrupt:
        #list(map(lambda p: p.join(), pool))
        print("++ [terminate]")


if __name__ == '__main__':
    start()
