"""Open the local live map gallery while keeping the existing lab available."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fly_brain_rc.dashboard.lab import run_lab

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('port must be between1024 and65535')
    run_lab(ROOT,port=args.port,open_path='/maps')
